from . import *
import src.net_core.darknet as darknet
import src.net_core.autoencoder as ae
import src.net_core.priornet as priornet
from function import *

class nolbo_singleObject(object):
    def __init__(self, config=None):
        #==========set parameters===========
        self._config = config
        self._enc_arc = config['encoder']
        self._dec_arc = config['decoder']
        self._prior_arc = config['prior']
        #==========declare networks=========
        self._encoder = None
        self._decoder = None
        self._classPriornet = None
        self._instPriornet = None
        #==========declare inputs===========
        self._inputImages = tf.placeholder(tf.float32, shape=np.concatenate([[None], self._enc_arc['inputImgDim']]))
        #==========declare predicted outputs===========
        self._classMeanPred, self._instMeanPred = None, None
        self._classLogVarPred, self._instLogVarPred = None, None
        self._zs = None
        self._outputImagesPred = None
        self._classMeanPrior, self._instMeanPrior = None,None
        self._classLogVarPrior, self._instLogVarPrior = None,None
        #==========declare parameters=========
        self._latentDim = 0
        #==========declare Ground Truth (GT) inputs for training==========
        self._outputImagesGT = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._dec_arc['outputImgDim']])))
        if self._config['class']:
            self._latentDim += self._config['zClassDim']
            self._classListGT = tf.placeholder(tf.float32, shape=([None, self._config['classDim']]))
        if self._config['inst']:
            self._latentDim += self._config['zInstDim']
            self._instListGT = tf.placeholder(tf.float32, shape=([None, self._config['instDim']]))
        if self._config['rot']:
            self._latentDim += 2*self._config['zRotDim'] # sine and cosine
            self._EulerAngleGT = tf.placeholder(tf.float32, shape=([None, 2*self._config['rotDim']]))
        # ==========declare parameters for training==============
        self._batchSize = None
        self._learningRate = tf.placeholder(tf.float32, shape=[])
        self._variables = None
        self._update_ops = None
        self._saver = None
        self._optimizer = None
        self._totalLoss = None

        # assumption of radian variance
        self._radVar = 0.03 ** 2

        self._buildNetwork()

        # init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # launch the session
        self._sess.run(init)

    def _buildNetwork(self):
        print "build network..."
        #=================== prior networks =====================
        self._classPriornet = priornet.priornet(
            inputDim=self._config['classDim'], outputDim=self._config['zClassDim'],
            hiddenLayerNum=self._prior_arc['hiddenLayerNum'],
            nameScope=self._config['nameScope'] + '-classPrior',
            training=self._config['isTraining'], trainable=self._prior_arc['trainable'],
            coreActivation=self._prior_arc['activation'],
            constLogVar=self._prior_arc['constLogVar'])
        self._instPriornet = priornet.priornet(
            inputDim=self._config['classDim'] + self._config['instDim'], outputDim=self._config['zInstDim'],
            hiddenLayerNum=self._prior_arc['hiddenLayerNum'],
            nameScope=self._config['nameScope'] + '-instPrior',
            training=self._config['isTraining'], trainable=self._prior_arc['trainable'],
            coreActivation=self._prior_arc['activation'],
            constLogVar=self._prior_arc['constLogVar'])
        self._classMeanPrior, self._classLogVarPrior = \
            self._classPriornet(self._classListGT)
        self._instMeanPrior, self._instLogVarPrior = \
            self._instPriornet(tf.concat([self._classListGT, self._instListGT], axis=-1))

        #=================== encoder and decoder ======================
        self._encoder = darknet.encoder(
            outputVectorDim= 2 * self._latentDim - self._config['zRotDim'], #multiply 2 as we want both mean and variance
            nameScope=self._config['nameScope'] + '-enc',
            trainable=self._enc_arc['trainable'],
            coreActivation=self._enc_arc['activation'],
            lastLayerActivation=None,
            lastLayerPooling=self._enc_arc['lastPool'],
            )
        self._decoder = ae.decoder(
            decoderName=self._config['nameScope'] + '-dec',
            architecture=self._dec_arc,
            )
        encOutput = self._encoder(self._inputImages)
        self._classMeanPred = encOutput[...,0:self._config['zClassDim']]
        self._instMeanPred = encOutput[...,self._config['zClassDim']:self._config['zClassDim']+self._config['zInstDim']]
        self._sinMeanPred = -1.0+2.0*tf.nn.sigmoid(encOutput[...,self._config['zClassDim']+self._config['zInstDim']:self._config['zClassDim']+self._config['zInstDim']+self._config['zRotDim']])
        self._cosMeanPred = -1.0+2.0*tf.nn.sigmoid(encOutput[...,self._config['zClassDim']+self._config['zInstDim']+self._config['zRotDim']:self._config['zClassDim']+self._config['zInstDim']+2*self._config['zRotDim']])

        logVarStart = self._config['zClassDim']+self._config['zInstDim']+2*self._config['zRotDim']
        self._classLogVarPred = encOutput[...,logVarStart:logVarStart+self._config['zClassDim']]
        self._instLogVarPred = encOutput[...,logVarStart+self._config['zClassDim']:logVarStart+self._config['zClassDim']+self._config['zInstDim']]
        self._radLogVarPred = encOutput[...,logVarStart+self._config['zClassDim']+self._config['zInstDim']:]
        # https://mathoverflow.net/questions/35260/resultant-probability-distribution-when-taking-the-cosine-of-gaussian-distribute
        # http://nbviewer.jupyter.org/gist/dougalsutherland/8513749
        self._Esinz = tf.exp(-tf.exp(self._radLogVarPred) / 2.0) * self._sinMeanPred
        self._Ecosz = tf.exp(-tf.exp(self._radLogVarPred) / 2.0) * self._cosMeanPred
        self._Varsinz = 0.5-0.5*tf.exp(-2.0*tf.exp(self._radLogVarPred))*(1.0-2.0*self._sinMeanPred*self._sinMeanPred)-tf.exp(-tf.exp(self._radLogVarPred))*self._sinMeanPred*self._sinMeanPred
        self._Varcosz = 0.5+0.5*tf.exp(-2.0*tf.exp(self._radLogVarPred))*(2.0*self._cosMeanPred*self._cosMeanPred-1.0)-tf.exp(-tf.exp(self._radLogVarPred))*self._cosMeanPred*self._cosMeanPred
        self._logVarsinz = tf.log(self._Varsinz)
        self._logVarcosz = tf.log(self._Varcosz)

        # normal output
        print 'get normal output'
        self._meanPred = tf.concat([self._classMeanPred, self._instMeanPred], axis=-1)
        self._logVarPred = tf.concat([self._classLogVarPred, self._instLogVarPred], axis=-1)
        self._zs = sampling(mu=self._meanPred, logVar=self._logVarPred)
        print self._zs.shape
        self._outputImagesPred = self._decoder(self._zs)


        if self._config['isTraining']:
            self._p, self._r = create_evaluation(xTarget=self._outputImagesGT, xPred=self._outputImagesPred)
            self._createLoss()
            self._setOptimizer()
        print "done!"

    def _createLoss(self):
        print "create loss..."
        self._binaryLoss = tf.reduce_mean(
            binary_loss(xPred=self._outputImagesPred, xTarget=self._outputImagesGT, gamma=0.60, b_range=False))
        self._cRLoss = tf.reduce_mean(
            regulizer_loss(
                z_mean=self._classMeanPrior, z_logVar=self._classLogVarPrior, dist_in_z_space=10.0))
        self._iRLoss = tf.reduce_mean(
            regulizer_loss(
                z_mean=self._instMeanPrior, z_logVar=self._instLogVarPrior, dist_in_z_space=5.0,
                class_input=self._classListGT))
        self._regulizerLoss = self._cRLoss + self._iRLoss
        AEISinGT = self._EulerAngleGT[..., :3]
        AEICosGT = self._EulerAngleGT[..., 3:]
        EsinzGT = tf.exp(-self._radVar / 2.0) * AEISinGT
        EcoszGT = tf.exp(-self._radVar / 2.0) * AEICosGT
        self._meanPrior = tf.concat([self._classMeanPrior, self._instMeanPrior, EsinzGT, EcoszGT], axis=-1)
        # https://mathoverflow.net/questions/35260/resultant-probability-distribution-when-taking-the-cosine-of-gaussian-distribute
        # http://nbviewer.jupyter.org/gist/dougalsutherland/8513749
        VarsinzGT = 0.5 - 0.5 * tf.exp(-2.0 * self._radVar) * (1.0 - 2.0 * AEISinGT * AEISinGT) - tf.exp(
            -self._radVar) * AEISinGT * AEISinGT
        VarcoszGT = 0.5 + 0.5 * tf.exp(-2.0 * self._radVar) * (2.0 * AEICosGT * AEICosGT - 1.0) - tf.exp(
            -self._radVar) * AEICosGT * AEICosGT
        logVarsinzGT = tf.log(VarsinzGT)
        logVarcoszGT = tf.log(VarcoszGT)
        self._logVarPrior = tf.concat([self._classLogVarPrior, self._instLogVarPrior, logVarsinzGT, logVarcoszGT],
                                      axis=-1)

        # self._nlbLoss = tf.reduce_mean(
        #     nlb_loss(mean=self._meanPred,
        #              logVar=self._logVarPred,mean_target=self._meanPrior, logVar_target=self._logVarPrior))

        self._classNLBLoss = tf.reduce_mean(nlb_loss(mean=self._classMeanPred, logVar=self._classLogVarPred,
                                                     mean_target=self._classMeanPrior,
                                                     logVar_target=self._classLogVarPrior))
        self._instNLBLoss = tf.reduce_mean(nlb_loss(mean=self._instMeanPred, logVar=self._instLogVarPred,
                                                    mean_target=self._instMeanPrior,
                                                    logVar_target=self._instLogVarPrior))
        self._classInstNLBLoss = self._classNLBLoss + self._instNLBLoss
        self._AEINLBLossSin = tf.reduce_mean(
            nlb_loss(mean=self._Esinz, logVar=self._logVarsinz, mean_target=EsinzGT, logVar_target=logVarsinzGT))
        self._AEINLBLossCos = tf.reduce_mean(
            nlb_loss(mean=self._Ecosz, logVar=self._logVarcosz, mean_target=EcoszGT, logVar_target=logVarcoszGT))
        self._AEINLBLoss = self._AEINLBLossSin + self._AEINLBLossCos

        self._sincosLoss = tf.reduce_mean(tf.reduce_sum(
            (self._sinMeanPred * self._sinMeanPred + self._cosMeanPred * self._cosMeanPred - 1.0)
            *
            (self._sinMeanPred * self._sinMeanPred + self._cosMeanPred * self._cosMeanPred - 1.0)
            , axis=-1))

        self._scmse = tf.reduce_mean(tf.reduce_sum(
            (self._sinMeanPred - AEISinGT) * (self._sinMeanPred - AEISinGT)
            +
            (self._cosMeanPred - AEICosGT) * (self._cosMeanPred - AEICosGT)
            , axis=-1))

        self._varmse = tf.reduce_mean(tf.reduce_sum(
            (self._radLogVarPred - tf.log(self._radVar)) * (self._radLogVarPred - tf.log(self._radVar))
            , axis=-1))

        self._nlbLoss = (
            (self._classNLBLoss + self._instNLBLoss)
            +
            0.1 * (self._AEINLBLoss)
        )

        self._totalLoss = (
            self._binaryLoss
            + self._nlbLoss + 0.1 * (self._cRLoss + self._iRLoss)
            + 100.0 * self._sincosLoss + 1000.0 * self._scmse + 100.0 * self._varmse
        )

    def _setOptimizer(self):
        print "set optimizer..."
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learningRate, momentum=0.90, use_nesterov=True)
        self._update_variables = tf.get_collection(key=None,scope=None)
        self._update_ops = tf.get_collection(key=None,scope=None)
        if self._enc_arc['trainable']:
            self._update_variables += self._encoder.allVariables
            self._update_ops += self._encoder.allUpdate_ops

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
            self._inputImages : batchDict['inputImages'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT : batchDict['classList'],
            self._instListGT : batchDict['instList'],
            self._EulerAngleGT : batchDict['AEIAngleSinCos']
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._binaryLoss, self._regulizerLoss, self._classInstNLBLoss, self._AEINLBLoss, self._scmse
        precisionRecall = self._p, self._r
        opt, loss, pr = self._sess.run([optimizer, lossList, precisionRecall], feed_dict=feed_dict)
        return loss, pr

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._config['nameScope'] + '_encoderCore.ckpt')
        self._encoder.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._config['nameScope'] + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.save(self._sess, eLastPath)
    def saveDecoder(self, savePath='./'):
        dPath = os.path.join(savePath, self._config['nameScope'] + '_decoder.ckpt')
        self._decoder.saver.save(self._sess, dPath)
    def savePriornet(self, savePath='./'):
        pCPath = os.path.join(savePath, self._config['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(savePath, self._config['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.save(self._sess, pCPath)
        self._instPriornet.saver.save(self._sess, pIPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
        self.saveDecoder(savePath)
        self.savePriornet(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._config['nameScope'] + '_encoderCore.ckpt')
        self._encoder.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._config['nameScope'] + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreDecoder(self, restorePath='./'):
        dPath = os.path.join(restorePath, self._config['nameScope'] + '_decoder.ckpt')
        self._decoder.saver.restore(self._sess, dPath)
    def restorePriornet(self, restorePath='./'):
        pCPath = os.path.join(restorePath, self._config['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(restorePath, self._config['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.restore(self._sess, pCPath)
        self._instPriornet.saver.restore(self._sess, pIPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)
        self.restoreDecoder(restorePath)
        self.restorePriornet(restorePath)

class nolbo_multiObjectTest(object):
    def __init__(self, config=None, nameScope='nolbo', isTraining=False):
        #==========set parameters===========
        if config==None:
            print 'set config first!'
            return
        self._config = config
        self._enc_arc = config['encoder']
        self._dec_arc = config['decoder']
        self._prior_arc = config['prior']
        self._nameScope = nameScope
        self._isTraining = isTraining
        #=============networks================
        self._encoder = None
        #=============inputs==============
        self._inputImages = None
        #=============outputs=============
        self._encOutput = None
        self._bboxHWXYPred = None
        self._objectnessPred = None
        self._meanPred = None
        self._logVarPred = None

        self._buildNetwork()

        # init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # launch the session
        self._sess.run(init)

    def _buildNetwork(self):
        # ========================== encoder ===========================
        self._latentDim = 0
        if self._config['class']:
            self._latentDim += self._config['zClassDim']
        if self._config['inst']:
            self._latentDim += self._config['zInstDim']
        if self._config['rot']:
            self._latentDim += 2 * self._config['zRotDim']  # sine and cosine
        self._inputImages = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._enc_arc['inputImgDim']])))
        self._encoder = darknet.encoder(
            outputVectorDim=self._config['predictorNumPerGrid']
                            * (self._config['bboxDim'] + 2 * self._latentDim - self._config['zRotDim']),
            # multiply 2 as we want both mean and variance
            nameScope=self._config['nameScope'] + '-enc',
            trainable=self._enc_arc['trainable'],
            coreActivation=self._enc_arc['activation'],
            lastLayerActivation=None,
            lastFilterNumList=self._enc_arc['lastFilterNumList'],
            lastKernelSizeList=self._enc_arc['lastKernelSizeList'],
            lastStridesList=self._enc_arc['lastStridesList'],
            lastLayerPooling=self._enc_arc['lastPool'],  # should be None
        )
        self._encOutput = self._encoder(self._inputImages)
        # bboxGT = (None,13,13,predictorNumPerGrid, bboxDim)
        self._bboxHWXYPred, self._objectnessPred, self._meanPred, self._logVarPred = [], [], [], []
        partStart = 0
        for predictorIndex in range(self._config['predictorNumPerGrid']):
            self._bboxHWXYPred.append(
                tf.nn.sigmoid(self._encOutput[..., partStart:partStart + self._config['bboxDim'] - 1]))
            partStart += self._config['bboxDim'] - 1
            self._objectnessPred.append(tf.nn.sigmoid(self._encOutput[..., partStart:partStart + 1]))
            partStart += 1
            # ==================== mu =========================
            classMeanPred = self._encOutput[..., partStart:partStart + self._config['zClassDim']]
            partStart += self._config['zClassDim']
            instMeanPred = self._encOutput[..., partStart:partStart + self._config['zInstDim']]
            partStart += self._config['zInstDim']
            sinMeanPred = -1.0 + 2.0 * tf.nn.sigmoid(
                self._encOutput[..., partStart:partStart + self._config['zRotDim']])
            partStart += self._config['zRotDim']
            cosMeanPred = -1.0 + 2.0 * tf.nn.sigmoid(
                self._encOutput[..., partStart:partStart + self._config['zRotDim']])
            partStart += self._config['zRotDim']
            # ================== log variance ==================
            classLogVarPred = self._encOutput[..., partStart:partStart + self._config['zClassDim']]
            partStart += self._config['zClassDim']
            instLogVarPred = self._encOutput[..., partStart:partStart + self._config['zInstDim']]
            partStart += self._config['zInstDim']
            radLogVarPred = self._encOutput[..., partStart:partStart + self._config['zRotDim']]
            partStart += self._config['zRotDim']
            # ================== concat =====================
            meanPred = tf.concat([classMeanPred, instMeanPred, sinMeanPred, cosMeanPred], axis=-1)
            logVarPred = tf.concat([classLogVarPred, instLogVarPred, radLogVarPred], axis=-1)
            self._meanPred.append(meanPred)
            self._logVarPred.append(logVarPred)
        self._bboxHWXYPred = tf.transpose(tf.stack(self._bboxHWXYPred), [1, 2, 3, 0, 4])
        self._objectnessPred = tf.transpose(tf.stack(self._objectnessPred), [1, 2, 3, 0, 4])
        self._meanPred = tf.transpose(tf.stack(self._meanPred), [1, 2, 3, 0, 4])
        self._logVarPred = tf.transpose(tf.stack(self._logVarPred), [1, 2, 3, 0, 4])
        print 'self._bboxHWXYs.shape', self._bboxHWXYPred.shape
        print 'self._objectnesses.shape', self._objectnessPred.shape
        print 'self._means.shape', self._meanPred.shape
        print 'self._logVars.shape', self._logVarPred.shape

    def getPred(self, inputImages):
        bboxHWXY, objectness, mean, logVar = self._sess.run(
            [self._bboxHWXYPred, self._objectnessPred, self._meanPred, self._logVarPred],
            feed_dict={self._inputImages:inputImages}
        )
        outputDict = {
            'bboxHWXY':bboxHWXY,
            'objectness':objectness,
            'mean':mean, 'logVar':logVar,
        }
        return outputDict

    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._encoder.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.restore(self._sess, eLastPath)
    def restore(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)

class nolbo_multiObject(object):
    def __init__(self, config=None, nameScope='nolbo', isTraining=True):
        #==========set parameters===========
        if config==None:
            print 'set config first!'
            return
        self._config = config
        self._enc_arc = config['encoder']
        self._dec_arc = config['decoder']
        self._prior_arc = config['prior']
        self._nameScope = nameScope
        self._isTraining = isTraining
        #==========declare networks==========
        self._encoder = None
        self._decoder = None
        self._classPriornet, self._instPriornet = None,None
        #==========declare inputs============
        self._inputImages = None
        #==========declare predicted outputs====================
        self._encOutput = None
        self._bboxHWXYPred = None
        self._objectnessPred = None
        self._meanPred = None
        self._logVarPred = None
        self._zs = None
        self._outputImagesPred = None
        #==========declare parameters ==========================
        self._latentDim = 0
        self._objMask = None
        #==========declare Ground Truth (GT) inputs for training============
        self._radVar = 0.05**2
        self._bboxHWXYGT = None
        self._objectnessGT = None
        self._outputImagesGT = None
        self._classListGT = None
        self._instListGT =None
        self._EulerAngleGT = None
        #==========declare parameters for training==============
        self._batchSize = None
        self._learningRate = None
        self._variables = None
        self._update_ops = None
        self._saver = None
        self._optimizer = None
        self._encOutChannelDim = None
        self._gridSize = None
        self._IOU = None
        self._totalLoss = None

        self._initTraining()
        self._buildNetwork()
        self._createLoss()
        self._setOptimizer()
        
        #init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        #initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        #launch the session
        self._sess.run(init)
        
    def _buildNetwork(self):
        print "build network..."
        # =================== prior networks =====================
        self._classPriornet = priornet.priornet(
            inputDim=self._config['classDim'], outputDim=self._config['zClassDim'],
            hiddenLayerNum=self._prior_arc['hiddenLayerNum'],
            nameScope=self._config['nameScope'] + '-classPrior',
            training=self._config['isTraining'], trainable=self._prior_arc['trainable'],
            coreActivation=self._prior_arc['activation'],
            constLogVar=self._prior_arc['constLogVar'])
        self._instPriornet = priornet.priornet(
            inputDim=self._config['classDim'] + self._config['instDim'], outputDim=self._config['zInstDim'],
            hiddenLayerNum=self._prior_arc['hiddenLayerNum'],
            nameScope=self._config['nameScope'] + '-instPrior',
            training=self._config['isTraining'], trainable=self._prior_arc['trainable'],
            coreActivation=self._prior_arc['activation'],
            constLogVar=self._prior_arc['constLogVar'])
        self._classMeanPrior, self._classLogVarPrior = \
            self._classPriornet(self._classListGT)
        self._instMeanPrior, self._instLogVarPrior = \
            self._instPriornet(tf.concat([self._classListGT, self._instListGT], axis=-1))
        #========================== VAE ===========================
        self._latentDim = 0
        if self._config['class']:
            self._latentDim += self._config['zClassDim']
        if self._config['inst']:
            self._latentDim += self._config['zInstDim']
        if self._config['rot']:
            self._latentDim += 2*self._config['zRotDim'] # sine and cosine
        self._inputImages = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._enc_arc['inputImgDim']])))
        self._encoder = darknet.encoder(
            outputVectorDim= self._config['predictorNumPerGrid']
                             *(self._config['bboxDim']+2*self._latentDim-self._config['zRotDim']), # multiply 2 as we want both mean and variance
            nameScope=self._config['nameScope'] + '-enc',
            trainable=self._enc_arc['trainable'],
            coreActivation=self._enc_arc['activation'],
            lastLayerActivation=None,
            lastFilterNumList=self._enc_arc['lastFilterNumList'],
            lastKernelSizeList=self._enc_arc['lastKernelSizeList'],
            lastStridesList=self._enc_arc['lastStridesList'],
            lastLayerPooling=self._enc_arc['lastPool'], #should be None
        )
        self._decoder = ae.decoder(
            decoderName=self._config['nameScope'] + '-dec',
            architecture=self._dec_arc,
        )
        self._encOutput = self._encoder(self._inputImages)
        #bboxGT = (None,13,13,predictorNumPerGrid, bboxDim)
        self._bboxHWXYPred, self._objectnessPred, self._meanPred, self._logVarPred = [],[],[],[]
        partStart = 0
        for predictorIndex in range(self._config['predictorNumPerGrid']):
            self._bboxHWXYPred.append(tf.nn.sigmoid(self._encOutput[...,partStart:partStart+self._config['bboxDim']-1]))
            partStart += self._config['bboxDim']-1
            self._objectnessPred.append(tf.nn.sigmoid(self._encOutput[...,partStart:partStart+1]))
            partStart += 1
            #==================== mu =========================
            classMeanPred = self._encOutput[...,partStart:partStart+self._config['zClassDim']]
            partStart += self._config['zClassDim']
            instMeanPred = self._encOutput[...,partStart:partStart+self._config['zInstDim']]
            partStart += self._config['zInstDim']
            sinMeanPred = -1.0+2.0*tf.nn.sigmoid(self._encOutput[...,partStart:partStart+self._config['zRotDim']])
            partStart += self._config['zRotDim']
            cosMeanPred = -1.0+2.0*tf.nn.sigmoid(self._encOutput[...,partStart:partStart+self._config['zRotDim']])
            partStart += self._config['zRotDim']
            #================== log variance ==================
            classLogVarPred = self._encOutput[...,partStart:partStart+self._config['zClassDim']]
            partStart += self._config['zClassDim']
            instLogVarPred = self._encOutput[...,partStart:partStart+self._config['zInstDim']]
            partStart += self._config['zInstDim']
            radLogVarPred = self._encOutput[...,partStart:partStart+self._config['zRotDim']]
            partStart += self._config['zRotDim']
            #================== concat =====================
            meanPred = tf.concat([classMeanPred,instMeanPred,sinMeanPred,cosMeanPred], axis=-1)
            logVarPred = tf.concat([classLogVarPred,instLogVarPred,radLogVarPred], axis=-1)
            self._meanPred.append(meanPred)
            self._logVarPred.append(logVarPred)
        self._bboxHWXYPred = tf.transpose(tf.stack(self._bboxHWXYPred), [1,2,3,0,4])
        self._objectnessPred = tf.transpose(tf.stack(self._objectnessPred), [1,2,3,0,4])
        self._meanPred = tf.transpose(tf.stack(self._meanPred), [1,2,3,0,4])
        self._logVarPred = tf.transpose(tf.stack(self._logVarPred), [1,2,3,0,4])
        print 'self._bboxHWXYs.shape',self._bboxHWXYPred.shape
        print 'self._objectnesses.shape',self._objectnessPred.shape
        print 'self._means.shape',self._meanPred.shape
        print 'self._logVars.shape',self._logVarPred.shape
        self._selectObjAndFracMeanLogVar()
        self._meanPredClassInst = tf.concat([self._classMeanPred, self._instMeanPred], axis=-1)
        self._logVarPredClassInst = tf.concat([self._classLogVarPred, self._instLogVarPred], axis=-1)
        self._zs = sampling(mu=self._meanPredClassInst, logVar=self._logVarPredClassInst)
        print 'self._zs.shape',self._zs.shape
        self._outputImagesPred = self._decoder(self._zs)
        print 'self._outputImagesPred.shape',self._outputImagesPred.shape

        self._p, self._r = create_evaluation(xTarget=self._outputImagesGT, xPred=self._outputImagesPred)

        objNum = tf.reduce_sum(self._objectnessGTWithIOU, axis=[1,2,3])
        noObjNum = tf.reduce_sum((1.0-self._objectnessGTWithIOU), axis=[1,2,3])
        self._objnessPrb = tf.reduce_mean(
            tf.reduce_sum(
                self._objectnessGTWithIOU
                *tf.reshape(self._objectnessPred, tf.shape(self._objectnessPred)[:-1]), axis=[1,2,3])/objNum)
        self._noobjnessPrb = tf.reduce_mean(
            tf.reduce_sum(
                (1.0-self._objectnessGTWithIOU)
                *(1.0-tf.reshape(self._objectnessPred, tf.shape(self._objectnessPred)[:-1])), axis=[1,2,3])/noObjNum)
        print "done!"
        
    def _createLoss(self):
        print "create loss..."
        diffBboxX = self._objMask*(self._bboxHWXYPredTile[...,2]-self._bboxHWXYGTTile[...,2])
        diffBboxY = self._objMask*(self._bboxHWXYPredTile[...,3]-self._bboxHWXYGTTile[...,3])
        diffBboxH = self._objMask*(tf.sqrt(self._bboxHWXYPredTile[...,0])-tf.sqrt(self._bboxHWXYGTTile[...,0]))
        diffBboxW = self._objMask*(tf.sqrt(self._bboxHWXYPredTile[...,1])-tf.sqrt(self._bboxHWXYGTTile[...,1]))

        # diffObj = self._objectnessGTWithIOU*(self._objectnessGTWithIOU-tf.reshape(self._objectnessPred, tf.shape(self._objectnessPred)[:-1]))
        # diffNoObj = (1.0-self._objectnessGTWithIOU)*(self._objectnessGTWithIOU-tf.reshape(self._objectnessPred, tf.shape(self._objectnessPred)[:-1]))
        # diffObj = tf.square(diffObj)
        # diffNoObj = tf.square(diffNoObj)
        diffObj = -self._objectnessGTWithIOU*tf.log(tf.reshape(self._objectnessPred, tf.shape(self._objectnessPred)[:-1])+1e-10)
        diffNoObj = -(1.0-self._objectnessGTWithIOU)*tf.log(1.0-tf.reshape(self._objectnessPred, tf.shape(self._objectnessPred)[:-1])+1e-10)

        # objTileShape = np.concatenate([[-1], self._objectnessPredTile.get_shape().as_list()[1:-1]])
        # diffObjPrObjxIOU = self._objMask*(tf.reshape(self._objectnessPredTile,objTileShape)-tf.reshape(self._objectnessGTTile,objTileShape))
        # diffNoObj = (1.0-self._objMask)*(tf.reshape(self._objectnessPredTile,objTileShape)-tf.reshape(self._objectnessGTTile,objTileShape))
        # diffObjPrObjxIOU = tf.square(diffObjPrObjxIOU)
        # diffNoObj = tf.square(diffNoObj)
        # diffObjPrObjxIOU = -self._objMask*tf.reshape(self._objectnessGTTile,objTileShape)*tf.log(tf.reshape(self._objectnessPredTile,objTileShape)+1e-10)
        # diffNoObj = -(1.0-self._objMask)*(1.0-tf.reshape(self._objectnessGTTile,objTileShape))*tf.log(1.0-tf.reshape(self._objectnessPredTile,objTileShape)+1e-10)

        self._bboxLoss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(diffBboxX)+tf.square(diffBboxY)+tf.square(diffBboxH)+tf.square(diffBboxW), axis=[1,2,3]))
        self._objLoss = tf.reduce_mean(tf.reduce_sum((diffObj), axis=[1,2,3]))
        self._noObjLoss = tf.reduce_mean(tf.reduce_sum((diffNoObj), axis=[1,2,3]))

        # voxelDimTotal = np.prod(self._outputImagesPred.get_shape().as_list()[1:])
        # outputImagesPredFlat = tf.reshape(self._outputImagesPred, (-1, voxelDimTotal))
        # voxelPred = outputImagesPredFlat*self._selectedObjectnessPred*self._selectedIOU
        self._binaryLoss = tf.reduce_mean(
            binary_loss(xPred=self._outputImagesPred, xTarget=self._outputImagesGT, gamma=0.60, b_range=False))
        self._cRLoss = tf.reduce_mean(
            regulizer_loss(
                z_mean=self._classMeanPrior, z_logVar=self._classLogVarPrior, dist_in_z_space=10.0))
        self._iRLoss = tf.reduce_mean(
            regulizer_loss(
                z_mean=self._instMeanPrior, z_logVar=self._instLogVarPrior, dist_in_z_space=5.0,
                class_input=self._classListGT))
        self._regulizerLoss = self._cRLoss + self._iRLoss
        AEISinGT = self._EulerAngleGT[..., :3]
        AEICosGT = self._EulerAngleGT[..., 3:]
        EsinzGT = tf.exp(-self._radVar / 2.0) * AEISinGT
        EcoszGT = tf.exp(-self._radVar / 2.0) * AEICosGT
        self._meanPrior = tf.concat([self._classMeanPrior, self._instMeanPrior, EsinzGT, EcoszGT], axis=-1)
        # https://mathoverflow.net/questions/35260/resultant-probability-distribution-when-taking-the-cosine-of-gaussian-distribute
        # http://nbviewer.jupyter.org/gist/dougalsutherland/8513749
        VarsinzGT = 0.5 - 0.5 * tf.exp(-2.0 * self._radVar) * (1.0 - 2.0 * AEISinGT * AEISinGT) - tf.exp(
            -self._radVar) * AEISinGT * AEISinGT
        VarcoszGT = 0.5 + 0.5 * tf.exp(-2.0 * self._radVar) * (2.0 * AEICosGT * AEICosGT - 1.0) - tf.exp(
            -self._radVar) * AEICosGT * AEICosGT
        logVarsinzGT = tf.log(VarsinzGT)
        logVarcoszGT = tf.log(VarcoszGT)
        self._logVarPrior = tf.concat([self._classLogVarPrior, self._instLogVarPrior, logVarsinzGT, logVarcoszGT],
                                      axis=-1)

        # self._nlbLoss = tf.reduce_mean(
        #     nlb_loss(mean=self._meanPred,
        #              logVar=self._logVarPred,mean_target=self._meanPrior, logVar_target=self._logVarPrior))

        self._classNLBLoss = tf.reduce_mean(nlb_loss(mean=self._classMeanPred, logVar=self._classLogVarPred,
                                                     mean_target=self._classMeanPrior, logVar_target=self._classLogVarPrior))
        self._instNLBLoss = tf.reduce_mean(nlb_loss(mean=self._instMeanPred, logVar=self._instLogVarPred,
                                                    mean_target=self._instMeanPrior, logVar_target=self._instLogVarPrior))
        self._classInstNLBLoss = self._classNLBLoss + self._instNLBLoss
        self._AEINLBLossSin = tf.reduce_mean(
            nlb_loss(mean=self._Esinz, logVar=self._logVarsinz, mean_target=EsinzGT, logVar_target=logVarsinzGT))
        self._AEINLBLossCos = tf.reduce_mean(
            nlb_loss(mean=self._Ecosz, logVar=self._logVarcosz, mean_target=EcoszGT, logVar_target=logVarcoszGT))
        self._AEINLBLoss = self._AEINLBLossSin + self._AEINLBLossCos

        self._sincosLoss = tf.reduce_mean(tf.reduce_sum(
            (self._sinMeanPred * self._sinMeanPred + self._cosMeanPred * self._cosMeanPred - 1.0)
            *
            (self._sinMeanPred * self._sinMeanPred + self._cosMeanPred * self._cosMeanPred - 1.0)
            , axis=-1))

        self._scmse = tf.reduce_mean(tf.reduce_sum(
            (self._sinMeanPred - AEISinGT) * (self._sinMeanPred - AEISinGT)
            +
            (self._cosMeanPred - AEICosGT) * (self._cosMeanPred - AEICosGT)
            , axis=-1))

        self._varmse = tf.reduce_mean(tf.reduce_sum(
            (self._radLogVarPred - tf.log(self._radVar)) * (self._radLogVarPred - tf.log(self._radVar))
            , axis=-1))

        self._nlbLoss = (
            (self._classNLBLoss + self._instNLBLoss)
            +
            0.1 * (self._AEINLBLoss)
        )

        self._totalLoss = ( # objnum:noobjnum = 1:150
            10.0*(5.0*self._bboxLoss + 10.0*self._objLoss + 0.5*self._noObjLoss)
            + self._binaryLoss
            + self._nlbLoss
            + 0.1 * (self._cRLoss + self._iRLoss)
            + 100.0 * self._sincosLoss + 1000.0 * self._scmse + 100.0 * self._varmse
        )

        print 'done!'
        
    def _setOptimizer(self):
        print "set optimizer..."
        # self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learningRate, momentum=0.90, use_nesterov=True)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learningRate)
        self._update_variables = tf.get_collection(key=None,scope=None)
        self._update_ops = tf.get_collection(key=None,scope=None)
        if self._enc_arc['trainable']:
            print 'update encoder...'
            self._update_variables += self._encoder.allVariables
            self._update_ops += self._encoder.allUpdate_ops
        if self._dec_arc['trainable']:
            print 'update decoder...'
            self._update_variables += self._decoder.variables
            self._update_ops += self._decoder.update_ops
        if self._prior_arc['trainable']:
            print 'update priorNet...'
            self._update_variables += self._classPriornet.variables + self._instPriornet.variables
            self._update_ops += self._classPriornet.update_ops + self._instPriornet.update_ops

        with tf.control_dependencies(self._update_ops):
            self._optimizer = self._optimizer.minimize(
                self._totalLoss,
                var_list=self._update_variables
            )
        print 'done!'
        
    def fit(self, batchDict):
        feed_dict = {
            self._learningRate : batchDict['learningRate'],
            self._inputImages : batchDict['inputImages'],
            self._bboxHWXYGT : batchDict['bboxHWXY'],
            self._objectnessGT : batchDict['objectness'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT : batchDict['classList'],
            self._instListGT : batchDict['instList'],
            self._EulerAngleGT : batchDict['EulerAngle']
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._bboxLoss, self._objLoss, self._noObjLoss, \
                   self._binaryLoss, self._regulizerLoss, self._classInstNLBLoss, self._AEINLBLoss, self._scmse,\
                    self._objnessPrb, self._noobjnessPrb
        precisionRecall = self._p, self._r
        opt, loss, pr = self._sess.run([optimizer, lossList, precisionRecall], feed_dict=feed_dict)
        return loss, pr

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._nameScope + '_encoderCore.ckpt')
        self._encoder.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.save(self._sess, eLastPath)
    def saveDecoder(self, savePath='./'):
        dPath = os.path.join(savePath, self._nameScope + '_decoder.ckpt')
        self._decoder.saver.save(self._sess, dPath)
    def savePriornet(self, savePath='./'):
        pCPath = os.path.join(savePath, self._config['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(savePath, self._config['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.save(self._sess, pCPath)
        self._instPriornet.saver.save(self._sess, pIPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
        self.saveDecoder(savePath)
        self.savePriornet(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._encoder.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreDecoder(self, restorePath='./'):
        dPath = os.path.join(restorePath, self._nameScope + '_decoder.ckpt')
        self._decoder.saver.restore(self._sess, dPath)
    def restorePriornet(self, restorePath='./'):
        pCPath = os.path.join(restorePath, self._config['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(restorePath, self._config['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.restore(self._sess, pCPath)
        self._instPriornet.saver.restore(self._sess, pIPath)
    def restoreNetwork(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)
        self.restoreDecoder(restorePath)
        self.restorePriornet(restorePath)

    def _selectObjAndFracMeanLogVar(self):
        self._getTiles()
        self._getObjMask()
#             Here, dynamic_partition is used instead of boolean_mask according to the following sites:
#             https://stackoverflow.com/questions/44380727/get-userwarning-while-i-use-tf-boolean-mask?noredirect=1&lq=1
#             https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation#35896823
#             mean = tf.boolean_mask(self._meanPredTile, tf.cast(self._objMask, tf.bool))
#             variance = tf.boolean_mask(self._variancePredTile, tf.cast(self._objMask, tf.bool))
        self._meanPred = tf.dynamic_partition(self._meanPredTile, tf.cast(self._objMask, tf.int32), 2)[1]
        self._logVarPred = tf.dynamic_partition(self._logVarPredTile, tf.cast(self._objMask, tf.int32), 2)[1]
        # self._selectedObjectnessPred = tf.dynamic_partition(self._objectnessPredTile, tf.cast(self._objMask, tf.int32), 2)[1]
        # selectedIOUShape = np.concatenate([[-1],self._IOU.get_shape().as_list()[1:],[1]])
        # self._selectedIOU = tf.dynamic_partition(tf.reshape(self._IOU,selectedIOUShape),tf.cast(self._objMask, tf.int32), 2)[1]

        self._classMeanPred = self._meanPred[..., 0:self._config['zClassDim']]
        self._instMeanPred = self._meanPred[...,self._config['zClassDim']:self._config['zClassDim']+self._config['zInstDim']]
        self._sinMeanPred = self._meanPred[...,self._config['zClassDim']+self._config['zInstDim']:self._config['zClassDim']+self._config['zInstDim']+self._config['zRotDim']]
        self._cosMeanPred = self._meanPred[...,self._config['zClassDim']+self._config['zInstDim']+self._config['zRotDim']:self._config['zClassDim']+self._config['zInstDim']+2*self._config['zRotDim']]

        self._classLogVarPred = self._logVarPred[..., 0:self._config['zClassDim']]
        self._instLogVarPred = self._logVarPred[...,self._config['zClassDim']:self._config['zClassDim']+self._config['zInstDim']]
        self._radLogVarPred = self._logVarPred[...,self._config['zClassDim']+self._config['zInstDim']:self._config['zClassDim']+self._config['zInstDim']+self._config['zRotDim']]

        # https://mathoverflow.net/questions/35260/resultant-probability-distribution-when-taking-the-cosine-of-gaussian-distribute
        # http://nbviewer.jupyter.org/gist/dougalsutherland/8513749
        self._Esinz = tf.exp(-tf.exp(self._radLogVarPred)/2.0)*self._sinMeanPred
        self._Ecosz = tf.exp(-tf.exp(self._radLogVarPred)/2.0)*self._cosMeanPred
        Varsinz=0.5-0.5*tf.exp(-2.0*tf.exp(self._radLogVarPred))*(
        1.0-2.0*self._sinMeanPred*self._sinMeanPred)-tf.exp(-tf.exp(self._radLogVarPred))*self._sinMeanPred*self._sinMeanPred
        Varcosz = 0.5+0.5*tf.exp(-2.0*tf.exp(self._radLogVarPred))*(
        2.0*self._cosMeanPred*self._cosMeanPred-1.0)-tf.exp(-tf.exp(self._radLogVarPred))*self._cosMeanPred*self._cosMeanPred
        self._logVarsinz = tf.log(Varsinz+1e-7)
        self._logVarcosz = tf.log(Varcosz+1e-7)

    def _initTraining(self):
        print "init training..."
        self._gridSize = [self._enc_arc['inputImgDim'][0]/2**self._config['maxPoolNum'], self._enc_arc['inputImgDim'][1]/2**self._config['maxPoolNum']]
        self._bboxHWXYGT = tf.placeholder(tf.float32, shape=([None, self._gridSize[0], self._gridSize[1], self._config['predictorNumPerGrid'], self._config['bboxDim']-1]))
        self._objectnessGT = tf.placeholder(tf.float32, shape=([None,self._gridSize[0], self._gridSize[1], self._config['predictorNumPerGrid'], 1]))
        self._outputImagesGT = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._dec_arc['outputImgDim']])))
        self._classListGT = tf.placeholder(tf.float32, shape=([None,self._config['classDim']]))
        self._instListGT = tf.placeholder(tf.float32, shape=([None, self._config['instDim']]))
        self._EulerAngleGT = tf.placeholder(tf.float32, shape=([None, 2*self._config['rotDim']]))
        self._learningRate = tf.placeholder(tf.float32, shape=[])
        offset = np.transpose(
            np.reshape(
                np.array(
                    [np.arange(self._gridSize[0])] * self._gridSize[1] * self._config['predictorNumPerGrid']),
                (self._config['predictorNumPerGrid'], self._gridSize[0], self._gridSize[1])
            )
            ,(1, 2, 0)
        )
        #offset for columns
        self._offset = tf.reshape(
            tf.constant(offset, dtype=tf.float32),
            [1, self._gridSize[0], self._gridSize[1], self._config['predictorNumPerGrid']]
        )
        print 'done!'
    
    def _getTiles(self):
        #tiles for prediction
        self._bboxHWXYPredTile = tf.tile(self._bboxHWXYPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._objectnessPredTile = tf.tile(self._objectnessPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._meanPredTile = tf.tile(self._meanPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._logVarPredTile = tf.tile(self._logVarPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._bboxHWXYGTTile = []
        self._objectnessGTTile = []
        for objInBboxGTIdx in range(self._config['predictorNumPerGrid']):
            for predictorIndex in range(self._config['predictorNumPerGrid']):
                self._bboxHWXYGTTile+=[self._bboxHWXYGT[:,:,:,objInBboxGTIdx:objInBboxGTIdx+1,:]]
                self._objectnessGTTile+=[self._objectnessGT[:,:,:,objInBboxGTIdx:objInBboxGTIdx+1,:]]
        self._bboxHWXYGTTile = tf.concat(self._bboxHWXYGTTile, axis=-2)
        self._objectnessGTTile = tf.concat(self._objectnessGTTile, axis=-2)
        
    def _getObjMask(self):
        print 'get ObjMask...'
        self._getIOU() #IOU = (None, 13,13, prdNPG*objNPG)
        self._objMaskArr = []
        for objIdx in range(self._config['predictorNumPerGrid']):
            #IOUPerObj = (None, 13,13, prdNPG)
            IOUPerObj = self._IOU[...,objIdx*self._config['predictorNumPerGrid']:(objIdx+1)*self._config['predictorNumPerGrid']]
            objectnessGTPerObj = self._objectnessGTTile[...,objIdx*self._config['predictorNumPerGrid']:(objIdx+1)*self._config['predictorNumPerGrid'],0]
            #maxPerObj = (None, 13,13)
            _, indices = tf.nn.top_k(IOUPerObj, k=self._config['predictorNumPerGrid'])
            # print "indices.shape",indices.shape
            indices = tf.reshape(indices, (-1,self._config['predictorNumPerGrid']))
            indices = tf.map_fn(tf.invert_permutation, indices)
            # print "indices.shape", indices.shape
            indices = tf.reshape(indices,(-1, self._gridSize[0],self._gridSize[1],self._config['predictorNumPerGrid']))
            # print "indices.shape", indices.shape
            # maxPerObj = tf.reduce_max(IOUPerObj, -1, keep_dims=True)
            # # maskPerObj = tf.cast((IOUPerObj>=maxPerObj and IOUPerObj!=0.0),tf.float32)
            # maskPerObj = tf.where(IOUPerObj>=maxPerObj, tf.ones_like(IOUPerObj), tf.zeros_like(IOUPerObj))
            maskPerObj = tf.cast(tf.where(indices<=0, tf.ones_like(indices), tf.zeros_like(indices)), tf.float32)
            maskPerObj = maskPerObj * objectnessGTPerObj
            self._objMaskArr+=[maskPerObj] # maskPerObj = (None,13,13,prdNPG)
            # print "maskPerObj.shape",maskPerObj.shape
        self._objectnessGTWithIOU = tf.stack(self._objMaskArr, axis=-1) #objectnessGTWithIOU=(None,13,13,prdNPG,objNPG)
        self._objectnessGTWithIOU = tf.reduce_sum(self._objectnessGTWithIOU, axis=-1) #objectnessGTWithIOU=(None,13,13,predN)
        self._objectnessGTWithIOU = tf.where(self._objectnessGTWithIOU>0, tf.ones_like(self._objectnessGTWithIOU), tf.zeros_like(self._objectnessGTWithIOU))
        self._objectnessGTWithIOU = tf.cast(self._objectnessGTWithIOU, tf.float32)
        print 'self._objectnessGTWithIOU',self._objectnessGTWithIOU.shape
        self._objMask = tf.concat(self._objMaskArr, axis=-1) #objMask=(None,13,13,prdNPG*objNPG)
        print 'self._objMask.shape',self._objMask.shape
        print 'done!'
    
    def _getIOU(self):
        print 'get IOU...'
        # bboxGT.shape = (None, 13,13,predictorNumPerGrid, bboxDim)
        # predictBbox.shape = (None, 13,13,predictorNumPerGrid, bboxDim)
        # bboxHWXY = [0:H, 1:W, 2:x, 3:y]
        # get bboxGTTile = (None, 13,13,predictorNumPerGrid*objNumPerGrid, bboxDim)
        # box = [colMin, rowMin, colMax, rowMax]
        #   = [x-W/2, y-H/2, x+W/2, y+H/2]
        offset_column = tf.tile(
            self._offset, [tf.shape(self._bboxHWXYGT)[0], 1,1,self._config['predictorNumPerGrid']])
        offset_row = tf.transpose(offset_column, (0, 2, 1, 3))
        print 'offset_row.shape', offset_row.shape
        boxGT = tf.stack([(self._bboxHWXYGTTile[:,:,:,:,2]+offset_column)/float(self._gridSize[1])
                          - self._bboxHWXYGTTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYGTTile[:,:,:,:,3]+offset_row)/float(self._gridSize[0])
                          - self._bboxHWXYGTTile[:,:,:,:,0]/2.0,
                          (self._bboxHWXYGTTile[:, :, :, :, 2] + offset_column) / float(self._gridSize[1])
                          + self._bboxHWXYGTTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYGTTile[:, :, :, :, 3] + offset_row) / float(self._gridSize[0])
                          + self._bboxHWXYGTTile[:,:,:,:,0]/2.0],
                         axis=-1) # (None, 13,13, prdNPG*objNPG, rcMinMax)
        print 'boxGT.shape', boxGT.shape
        boxPr = tf.stack([(self._bboxHWXYPredTile[:,:,:,:,2]+offset_column)/float(self._gridSize[1])
                          - self._bboxHWXYPredTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYPredTile[:,:,:,:,3]+offset_row)/float(self._gridSize[0])
                          - self._bboxHWXYPredTile[:,:,:,:,0]/2.0,
                          (self._bboxHWXYPredTile[:, :, :, :, 2] + offset_column) / float(self._gridSize[1])
                          + self._bboxHWXYPredTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYPredTile[:, :, :, :, 3] + offset_row) / float(self._gridSize[0])
                          + self._bboxHWXYPredTile[:,:,:,:,0]/2.0],
                         axis=-1) # (None, 13,13, prdNPG*objNPG, rcMinMax)
        print 'boxPr.shape', boxPr.shape
        #get leftup point and rightdown point of intersection
        leftUp = tf.maximum(boxGT[...,:2], boxPr[...,:2])
        print 'leftUp.shape', leftUp.shape
        rightDown = tf.minimum(boxGT[...,2:], boxPr[...,2:])
        print 'rightDown.shape', rightDown.shape
        #get intersection Areapyth
        intersection = tf.maximum(0.0, rightDown - leftUp)
        intersectionArea = intersection[...,0] * intersection[...,1]
        #get boxGTArea and boxPrArea
        boxGTArea = (boxGT[...,2] - boxGT[...,0]) * (boxGT[...,3] - boxGT[...,1])
        boxPrArea = (boxPr[...,2] - boxPr[...,0]) * (boxPr[...,3] - boxPr[...,1])
        unionArea = tf.maximum(boxGTArea + boxPrArea - intersectionArea, 1e-9)
        self._IOU = tf.clip_by_value(intersectionArea/unionArea, 0.0, 1.0) #IOU = (None, 13,13, prdNPG*objNPG)
        print 'self._IOU.shape', self._IOU.shape
        print 'done!'




