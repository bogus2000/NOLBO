from function import *
import src.net_core.autoencoder as ae
import src.net_core.priornet as priornet

class VAE3D(object):
    def __init__(self, network_architecture):
        #============= set network parameters ==============
        self._net_arc = network_architecture
        self._enc_arc = network_architecture['encoder']
        self._dec_arc = network_architecture['decoder']
        self._prior_arc = network_architecture['prior']

        #encoder input
        self._inputImages = tf.placeholder(tf.float32, shape=np.concatenate([[None], self._enc_arc['inputImgDim']]))

        #assumption of radian variance
        self._radVar = 0.05**2

        #priornet input
        self._latentDim = 0
        if self._net_arc['class']:
            self._latentDim += self._net_arc['zClassDim']
            self._classListGT = tf.placeholder(tf.float32, shape=([None, self._net_arc['classDim']]))
        if self._net_arc['inst']:
            self._latentDim += self._net_arc['zInstDim']
            self._instListGT = tf.placeholder(tf.float32, shape=([None, self._net_arc['instDim']]))
        if self._net_arc['rot']:
            self._latentDim += self._net_arc['zRotDim']
            self._EulerAngleGT = tf.placeholder(tf.float32, shape=([None, 2*self._net_arc['rotDim']]))

        #decoder input
        self._z = None
        #decoder output
        self._outputImages = None
        #decoder output GT
        self._outputImagesGT = tf.placeholder(tf.float32, shape=np.concatenate([[None], self._enc_arc['inputImgDim']]))

        #for training
        self._learningRate = tf.placeholder(tf.float32, shape=[])

        self._buildNetwork()

        # init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        # initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # launch the session
        self._sess.run(init)

    def _buildNetwork(self):
        print 'build network...'
        #=========== prior networks ============
        self._classPriornet = priornet.priornet(
            inputDim=self._net_arc['classDim'], outputDim=self._net_arc['zClassDim'],
            hiddenLayerNum=self._prior_arc['hiddenLayerNum'],
            nameScope=self._net_arc['nameScope'] + '-classPrior',
            training=self._net_arc['isTraining'], trainable=self._prior_arc['trainable'],
            coreActivation=self._prior_arc['activation'],
            constLogVar=self._prior_arc['constLogVar'])
        self._instPriornet = priornet.priornet(
            inputDim=self._net_arc['classDim'] + self._net_arc['instDim'], outputDim=self._net_arc['zInstDim'],
            hiddenLayerNum=self._prior_arc['hiddenLayerNum'],
            nameScope=self._net_arc['nameScope'] + '-instPrior',
            training=self._net_arc['isTraining'], trainable=self._prior_arc['trainable'],
            coreActivation=self._prior_arc['activation'],
            constLogVar=self._prior_arc['constLogVar'])
        self._classMeanPrior, self._classLogVarPrior = \
            self._classPriornet(self._classListGT)
        self._instMeanPrior, self._instLogVarPrior = \
            self._instPriornet(tf.concat([self._classListGT, self._instListGT], axis=-1))

        #============ encoder and decoder ============
        self._encoder = ae.encoder(self._enc_arc, encoderName='encoder3D')
        self._decoder = ae.decoder(self._dec_arc, decoderName='nolbo-dec')
        #====== input voxel value range : [0,1] -> [-1,2]
        encOutput = self._encoder(2.0*self._inputImages-1.0)
        self._classMeanPred = encOutput[...,0:self._net_arc['zClassDim']]
        self._instMeanPred = encOutput[...,self._net_arc['zClassDim']:self._net_arc['zClassDim']+self._net_arc['zInstDim']]
        self._sinMeanPred = -1.0+2.0*tf.nn.sigmoid(encOutput[...,self._net_arc['zClassDim']+self._net_arc['zInstDim']:self._net_arc['zClassDim']+self._net_arc['zInstDim']+self._net_arc['zRotDim']])
        self._cosMeanPred = -1.0+2.0*tf.nn.sigmoid(encOutput[...,self._net_arc['zClassDim']+self._net_arc['zInstDim']+self._net_arc['zRotDim']:self._net_arc['zClassDim']+self._net_arc['zInstDim']+2*self._net_arc['zRotDim']])

        logVarStart = self._net_arc['zClassDim']+self._net_arc['zInstDim']+2*self._net_arc['zRotDim']
        self._classLogVarPred = encOutput[...,logVarStart:logVarStart+self._net_arc['zClassDim']]
        self._instLogVarPred = encOutput[...,logVarStart+self._net_arc['zClassDim']:logVarStart+self._net_arc['zClassDim']+self._net_arc['zInstDim']]
        self._radLogVarPred = encOutput[...,logVarStart+self._net_arc['zClassDim']+self._net_arc['zInstDim']:]
        # https://mathoverflow.net/questions/35260/resultant-probability-distribution-when-taking-the-cosine-of-gaussian-distribute
        # http://nbviewer.jupyter.org/gist/dougalsutherland/8513749
        self._Esinz = tf.exp(-tf.exp(self._radLogVarPred) / 2.0) * self._sinMeanPred
        self._Ecosz = tf.exp(-tf.exp(self._radLogVarPred) / 2.0) * self._cosMeanPred
        self._Varsinz = 0.5-0.5*tf.exp(-2.0*tf.exp(self._radLogVarPred))*(1.0-2.0*self._sinMeanPred*self._sinMeanPred)-tf.exp(-tf.exp(self._radLogVarPred))*self._sinMeanPred*self._sinMeanPred
        self._Varcosz = 0.5+0.5*tf.exp(-2.0*tf.exp(self._radLogVarPred))*(2.0*self._cosMeanPred*self._cosMeanPred-1.0)-tf.exp(-tf.exp(self._radLogVarPred))*self._cosMeanPred*self._cosMeanPred
        self._logVarsinz = tf.log(self._Varsinz)
        self._logVarcosz = tf.log(self._Varcosz)

        #get sincos
        print 'get sincos'
        SAEI = sampling(mu=self._Esinz, logVar=self._logVarsinz)
        CAEI = sampling(mu=self._Ecosz, logVar=self._logVarcosz)
        # SAEI = sampling(mu=EsinzGT, logVar=VarsinzGT)
        # CAEI = sampling(mu=EcoszGT, logVar=VarcoszGT)
        # SAEI = AEISinGT
        # CAEI = AEICosGT
        # rotate inversely with negative angles to get a correct coordinate
        SA,SE,SI = -SAEI[...,0], -SAEI[...,1], -SAEI[...,2]
        CA,CE,CI = CAEI[...,0], CAEI[...,1], CAEI[...,2]
        SA = tf.tile(tf.reshape(SA, (-1,1,1,1,1)), (1,64,64,64,1))
        SE = tf.tile(tf.reshape(SE, (-1, 1, 1, 1, 1)), (1, 64, 64, 64, 1))
        SI = tf.tile(tf.reshape(SI, (-1, 1, 1, 1, 1)), (1, 64, 64, 64, 1))
        CA = tf.tile(tf.reshape(CA, (-1, 1, 1, 1, 1)), (1, 64, 64, 64, 1))
        CE = tf.tile(tf.reshape(CE, (-1, 1, 1, 1, 1)), (1, 64, 64, 64, 1))
        CI = tf.tile(tf.reshape(CI, (-1, 1, 1, 1, 1)), (1, 64, 64, 64, 1))
        print CA.shape

        # normal output
        print 'get normal output'
        self._meanPred = tf.concat([self._classMeanPred, self._instMeanPred], axis=-1)
        self._logVarPred = tf.concat([self._classLogVarPred, self._instLogVarPred], axis=-1)
        self._zs = sampling(mu=self._meanPred, logVar=self._logVarPred)
        print self._zs.shape
        self._outputImages = self._decoder(self._zs)

        batchSize = tf.shape(self._outputImages)[0]
        #set xyz coordinate
        print 'set xyz coordinate'
        yTile = np.reshape(np.array([np.arange(64)] * 64), (64, 64, 1))
        xTile = np.transpose(yTile, (1, 0, 2))
        yTile = np.reshape(np.tile(yTile, (1, 1, 64)), (1, 64, 64, 64, 1))
        xTile = np.reshape(np.tile(xTile, (1, 1, 64)), (1, 64, 64, 64, 1))
        zTile = np.reshape(np.transpose(xTile, (0, 3, 2, 1, 4)), (1, 64, 64, 64, 1))
        xTile = tf.tile(tf.cast(tf.constant(xTile), tf.float32) - 63.0/2.0, [batchSize, 1, 1, 1, 1])
        yTile = tf.tile(tf.cast(tf.constant(yTile), tf.float32) - 63.0/2.0, [batchSize, 1, 1, 1, 1])
        zTile = tf.tile(tf.cast(tf.constant(zTile), tf.float32) - 63.0/2.0, [batchSize, 1, 1, 1, 1])
        print xTile.shape

        # get rotated coordinate
        print 'get rotated coordinate'
        xRot = (CA*CI+SA*SE*SI)*xTile + (-SA*CI+CA*SE*SI)*yTile + (CE*SI)*zTile
        yRot = (SA*CE)*xTile + (CA*CE)*yTile + (-SE)*zTile
        zRot = (-CA*SI+SA*SE*CI)*xTile + (SA*SI+CA*SE*CI)*yTile + (CE*CI)*zTile

        idxI = tf.cast(xRot + 63.0/2.0 + 0.5, tf.int32)
        idxJ = tf.cast(yRot + 63.0/2.0 + 0.5, tf.int32)
        idxK = tf.cast(zRot + 63.0/2.0 + 0.5, tf.int32)
        idx = tf.concat([idxI,idxJ,idxK], axis=-1)
        idx = tf.where(idx>0, idx, tf.zeros_like(idx))
        idx = tf.where(idx<64, idx, tf.zeros_like(idx))
        print idx.shape

        #get batch index
        bIdx = tf.transpose(tf.reshape(tf.tile(tf.range(batchSize),[64*64*64*1]), [1,64,64,64,batchSize]))

        #get final index
        indices = tf.concat([bIdx, idx], axis=-1)

        #get rotated voxels
        self._outputImagesRotated = tf.gather_nd(params=self._outputImages, indices=indices)
        print self._outputImagesRotated.shape

        #final voxels
        # self._outputImages = tf.concat([self._outputImages, self._outputImagesRotated], axis=0)

        if self._net_arc['isTraining']:
            self._p, self._r = create_evaluation(xTarget=self._outputImagesGT, xPred=self._outputImages)
            self._createLoss()
            self._setOptimizer()

        print "done!"

    def _createLoss(self):
        print "create loss..."
        self._binaryLoss = tf.reduce_mean(
            binary_loss(xPred=self._outputImages, xTarget=self._outputImagesGT, gamma=0.60, b_range=False))
        self._cRLoss = tf.reduce_mean(
            regulizer_loss(
                z_mean=self._classMeanPrior, z_logVar=self._classLogVarPrior,dist_in_z_space=10.0))
        self._iRLoss = tf.reduce_mean(
            regulizer_loss(
                z_mean=self._instMeanPrior, z_logVar=self._instLogVarPrior,dist_in_z_space=5.0, class_input=self._classListGT))
        self._regulizerLoss = self._cRLoss + self._iRLoss
        AEISinGT = self._EulerAngleGT[...,:3]
        AEICosGT = self._EulerAngleGT[...,3:]
        EsinzGT = tf.exp(-self._radVar/2.0)*AEISinGT
        EcoszGT = tf.exp(-self._radVar/2.0)*AEICosGT
        self._meanPrior = tf.concat([self._classMeanPrior, self._instMeanPrior, EsinzGT, EcoszGT], axis=-1)
        # https://mathoverflow.net/questions/35260/resultant-probability-distribution-when-taking-the-cosine-of-gaussian-distribute
        # http://nbviewer.jupyter.org/gist/dougalsutherland/8513749
        VarsinzGT = 0.5-0.5*tf.exp(-2.0*self._radVar)*(1.0-2.0*AEISinGT*AEISinGT)-tf.exp(-self._radVar)*AEISinGT*AEISinGT
        VarcoszGT = 0.5+0.5*tf.exp(-2.0*self._radVar)*(2.0*AEICosGT*AEICosGT-1.0)-tf.exp(-self._radVar)*AEICosGT*AEICosGT
        logVarsinzGT = tf.log(VarsinzGT)
        logVarcoszGT = tf.log(VarcoszGT)
        self._logVarPrior = tf.concat([self._classLogVarPrior, self._instLogVarPrior, logVarsinzGT, logVarcoszGT],axis=-1)

        # self._nlbLoss = tf.reduce_mean(
        #     nlb_loss(mean=self._meanPred,
        #              logVar=self._logVarPred,mean_target=self._meanPrior, logVar_target=self._logVarPrior))

        self._classNLBLoss = tf.reduce_mean(nlb_loss(mean=self._classMeanPred, logVar=self._classLogVarPred,
                                                     mean_target=self._classMeanPrior, logVar_target=self._classLogVarPrior))
        self._instNLBLoss = tf.reduce_mean(nlb_loss(mean=self._instMeanPred, logVar=self._instLogVarPred,
                                                    mean_target=self._instMeanPrior, logVar_target=self._instLogVarPrior))
        self._classInstNLBLoss = self._classNLBLoss + self._instNLBLoss
        self._AEINLBLossSin = tf.reduce_mean(nlb_loss(mean=self._Esinz, logVar=self._logVarsinz, mean_target=EsinzGT, logVar_target=logVarsinzGT))
        self._AEINLBLossCos = tf.reduce_mean(nlb_loss(mean=self._Ecosz, logVar=self._logVarcosz, mean_target=EcoszGT, logVar_target=logVarcoszGT))
        self._AEINLBLoss = self._AEINLBLossSin + self._AEINLBLossCos

        self._sincosLoss = tf.reduce_mean(tf.reduce_sum(
            (self._sinMeanPred*self._sinMeanPred + self._cosMeanPred*self._cosMeanPred - 1.0)
            *
            (self._sinMeanPred*self._sinMeanPred + self._cosMeanPred*self._cosMeanPred - 1.0)
        , axis=-1))

        self._scmse = tf.reduce_mean(tf.reduce_sum(
            (self._sinMeanPred-AEISinGT)*(self._sinMeanPred-AEISinGT)
            +
            (self._cosMeanPred-AEICosGT)*(self._cosMeanPred-AEICosGT)
            , axis=-1))

        self._varmse = tf.reduce_mean(tf.reduce_sum(
            (self._radLogVarPred - tf.log(self._radVar))*(self._radLogVarPred - tf.log(self._radVar))
        , axis=-1))

        self._nlbLoss = (
            (self._classNLBLoss + self._instNLBLoss)
            +
            0.1*(self._AEINLBLoss)
        )

        self._totalLoss = (
            self._binaryLoss
            + self._nlbLoss + 0.1*(self._cRLoss+self._iRLoss)
            + 100.0*self._sincosLoss + 1000.0*self._scmse + 100.0*self._varmse
        )

    def _setOptimizer(self):
        print "set optimizer..."
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learningRate)
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learningRate, momentum=0.9, use_nesterov=True)
        self._update_variables = tf.get_collection(key=None,scope=None)
        self._update_ops = tf.get_collection(key=None,scope=None)
        if self._enc_arc['trainable']:
            self._update_variables += self._encoder.variables
            self._update_ops += self._encoder.update_ops
        if self._dec_arc['trainable']:
            self._update_variables += self._decoder.variables
            self._update_ops += self._decoder.update_ops
        if self._prior_arc['trainable']:
            self._update_variables += self._classPriornet.variables + self._instPriornet.variables
            self._update_ops += self._classPriornet.update_ops + self._instPriornet.update_ops

        with tf.control_dependencies(self._update_ops):
            self._optimizer = self._optimizer.minimize(
                self._totalLoss,
                var_list=self._update_variables
            )

    def fit(self, batchDict):
        feed_dict = {
            self._learningRate: batchDict['learningRate'],
            self._inputImages: batchDict['inputImages'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT: batchDict['classList'],
            self._instListGT: batchDict['instList'],
            self._EulerAngleGT: batchDict['EulerAngleSinCos']
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._binaryLoss, self._regulizerLoss, self._classInstNLBLoss, self._AEINLBLoss, self._scmse
        precisionRecall = self._p, self._r
        opt, loss, pr = self._sess.run([optimizer, lossList, precisionRecall], feed_dict=feed_dict)
        return loss, pr

    def saveEncoder(self, savePath='./'):
        ePath = os.path.join(savePath, self._net_arc['nameScope'] + '_encoderCore.ckpt')
        self._encoder.saver.save(self._sess, ePath)
    def saveDecoder(self, savePath='./'):
        dPath = os.path.join(savePath, self._net_arc['nameScope'] + '_decoder.ckpt')
        self._decoder.saver.save(self._sess, dPath)
    def savePriornet(self, savePath='./'):
        pCPath = os.path.join(savePath, self._net_arc['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(savePath, self._net_arc['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.save(self._sess, pCPath)
        self._instPriornet.saver.save(self._sess, pIPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoder(savePath)
        self.saveDecoder(savePath)
        self.savePriornet(savePath)
    def restoreEncoder(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._net_arc['nameScope'] + '_encoderCore.ckpt')
        self._encoder.saver.restore(self._sess, eCorePath)
    def restoreDecoder(self, restorePath='./'):
        dPath = os.path.join(restorePath, self._net_arc['nameScope'] + '_decoder.ckpt')
        self._decoder.saver.restore(self._sess, dPath)
    def restorePriornet(self, restorePath='./'):
        pCPath = os.path.join(restorePath, self._net_arc['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(restorePath, self._net_arc['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.restore(self._sess, pCPath)
        self._instPriornet.saver.restore(self._sess, pIPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoder(restorePath)
        self.restoreDecoder(restorePath)
        self.restorePriornet(restorePath)
