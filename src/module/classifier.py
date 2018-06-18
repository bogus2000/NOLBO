from . import *
import src.net_core.darknet as darknet

class darknet_classifier(object):
    def __init__(
            self,
            dataPath='./',
            nameScope='nolbo',
            imgSize = (None,None), batchSize = 64, learningRate = 0.0001,
            classNum = 40,
            coreLayerActivation = tf.nn.elu,
            lastLayerActivation = tf.nn.softmax):
        self._imgList = None
        self._imgClassList = None
        self._dataPath = dataPath
        self._nameScope = nameScope
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
        self._coreAct = coreLayerActivation
        self._lastAct = lastLayerActivation
        self._classNum = classNum
        self.variables = None
        self.update_ops = None
        self._inputImg = None
        self._outputClass = None
        self._outputClassGT = None
        self._optimizer = None
        self._loss = None
        self._buildNetwork()
        self._createLossAndOptimizer()
        self._createEvaluation()
        #init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # init = tf.initialize_all_variables()
        #launch the session
        self._sess.run(init)

    def _buildNetwork(self):
        print "build network..."
        self._inputImg = tf.placeholder(tf.float32, shape=(None, self._imgSize[0], self._imgSize[1], 3))
        self._outputClassGT = tf.placeholder(tf.float32, shape=(None, self._classNum))
        self._classifier = darknet.encoder(
            outputVectorDim=self._classNum,
            coreActivation=self._coreAct,
            lastLayerActivation=self._lastAct,
            lastLayerPooling='average',
            nameScope=self._nameScope+'-enc')
        self._outputClass = self._classifier(self._inputImg)
        print "done!"
    def _createLossAndOptimizer(self):
        print "create loss and optimizer..."
        self._lr = tf.placeholder(tf.float32, shape=[])
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        # self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.90, use_nesterov=True)
        # self._optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self._lr)
        # self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._lr)
        # self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._lr)
        # self._loss = tf.reduce_mean(
        #     tf.losses.softmax_cross_entropy(logits=self._outputClass, onehot_labels=self._outputClassGT))

        # self._loss = tf.reduce_sum(
        #     tf.losses.softmax_cross_entropy(onehot_labels=self._outputClassGT, logits=self._outputClass))

        # self._loss = tf.reduce_mean(-tf.reduce_sum(self._outputClassGT * tf.log(self._outputClass+1e-8), reduction_indices=1))
        self._loss = categorical_crossentropy(gt=self._outputClassGT, pred=self._outputClass)
        # self._loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._outputClassGT, logits=self._outputClass))
        with tf.control_dependencies(self._classifier.allUpdate_ops):
            self._optimizer = self._optimizer.minimize(
                self._loss, var_list = self._classifier.allVariables
            )
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self._optimizer = self._optimizer.minimize(
        #         self._loss
        #     )
        print "done!"
    def _createEvaluation(self):
        prediction = tf.argmax(self._outputClass, -1)
        gt = tf.argmax(self._outputClassGT, -1)
        equality = tf.equal(prediction, gt)
        self._acc = tf.reduce_mean(tf.cast(equality, tf.float32))
        self._top5Acc = tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=self._outputClass,
                    targets=gt,
                    k=5),
                tf.float32))

    def fit(self, batchDict):
        feed_dict = {
            self._inputImg : batchDict['inputImages'],
            self._outputClassGT : batchDict['classIndexList'],
            self._lr : batchDict['learningRate']
        }
        opt, lossResult, accResult, top5AccResult = self._sess.run([self._optimizer, self._loss, self._acc, self._top5Acc], feed_dict=feed_dict)
        return lossResult, accResult, top5AccResult

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.save(self._sess, eLastPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)


class RenderforCNN_classifier(object):
    def __init__(self, classNum, instNum, rotDim, coreActivation=tf.nn.elu, nameScope='nolbo'):
        self._classNum = classNum
        self._instNum = instNum
        self._rotDim = rotDim
        self._coreAct = coreActivation
        self._nameScope = nameScope
        #inputs
        self._inputImages = None
        #outputs
        self._classPred = None
        self._instPred = None
        self._azimuthPred = None
        self._elevationPred = None
        self._in_plane_rotPred = None
        #output ground truth
        self._classGT = None
        self._instGT = None
        self._azimuthGT = None
        self._elevationGT = None
        self._in_plane_rotGT = None
        #parameters for training
        self._learningRate = None

        self._initVariables()
        self._buildNetwork()
        self._createEvaluation()
        self._createLoss()
        self._setOptimizer()

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

    def _initVariables(self):
        # self._inputImages = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self._learningRate = tf.placeholder(tf.float32, shape=[])
        self._classGT = tf.placeholder(tf.float32, shape=(None, self._classNum))
        self._instGT = tf.placeholder(tf.float32, shape=(None, self._instNum))
        self._azimuthGT = tf.placeholder(tf.float32, shape=(None, self._rotDim))
        self._elevationGT = tf.placeholder(tf.float32, shape=(None, self._rotDim))
        self._in_plane_rotGT = tf.placeholder(tf.float32, shape=(None, self._rotDim))
    def _buildNetwork(self):
        print "build network..."
        self._inputImages = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self._classifier = darknet.encoder(
            outputVectorDim = self._classNum + self._instNum + 3*self._rotDim,
            coreActivation=self._coreAct,
            lastLayerActivation=None,
            lastLayerPooling='average',
            nameScope=self._nameScope + '-enc')
        self._encoderOutput = self._classifier(self._inputImages)
        self._classPred = tf.nn.softmax(self._encoderOutput[...,:self._classNum])
        self._instPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum:self._classNum+self._instNum])
        self._azimuthPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum+self._instNum:self._classNum+self._instNum+self._rotDim])
        self._elevationPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum+self._instNum+self._rotDim:self._classNum+self._instNum+2*self._rotDim])
        self._in_plane_rotPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum+self._instNum+2*self._rotDim:])
        print "done!"

    def _createLoss(self):
        print "create loss..."
        # self._classLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._classGT, logits=self._classPred))
        # self._instLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._instGT, logits=self._instPred))
        # self._azimuthLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._azimuthGT, logits=self._azimuthPred))
        # self._elevationLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._elevationGT, logits=self._elevationPred))
        # self._in_plane_rotLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._in_plane_rotGT, logits=self._in_plane_rotPred))
        self._classLoss = categorical_crossentropy(gt=self._classGT, pred=self._classPred)
        self._instLoss = categorical_crossentropy(gt=self._instGT, pred=self._instPred)
        self._azimuthLoss = categorical_crossentropy(gt=self._azimuthGT, pred=self._azimuthPred)
        self._elevationLoss = categorical_crossentropy(gt=self._elevationGT, pred=self._elevationPred)
        self._in_plane_rotLoss = categorical_crossentropy(gt=self._in_plane_rotGT, pred=self._in_plane_rotPred)
        self._loss = self._classLoss + self._instLoss + \
                     (self._azimuthLoss + self._elevationLoss + self._in_plane_rotLoss)
        print "done!"

    def _setOptimizer(self):
        print "set optimizer..."
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learningRate)
        # self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learningRate)
        # self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learningRate)
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learningRate, momentum=0.9, use_nesterov=True)
        with tf.control_dependencies(self._classifier.allUpdate_ops):
            self._optimizer = self._optimizer.minimize(
                self._loss, var_list = self._classifier.allVariables
            )
        print "done!"

    def _createEvaluation(self):
        classPrediction = tf.argmax(self._classPred, -1)
        classGT = tf.argmax(self._classGT, -1)
        classEquality = tf.equal(classPrediction, classGT)
        self._classAcc = tf.reduce_mean(tf.cast(classEquality, tf.float32))
        instPrediction = tf.argmax(self._instPred, -1)
        instGT = tf.argmax(self._instGT, -1)
        instEquality = tf.equal(instPrediction, instGT)
        self._instAcc = tf.reduce_mean(tf.cast(instEquality, tf.float32))

        azimuthPrediction = tf.argmax(self._azimuthPred, -1)
        azimuthGT = tf.argmax(self._azimuthGT, -1)
        azimuthEquality = tf.equal(azimuthPrediction, azimuthGT)
        self._azimuthAcc = tf.reduce_mean(tf.cast(azimuthEquality, tf.float32))
        self._azimuthTop5Acc = tf.reduce_mean(tf.cast(
            tf.nn.in_top_k(predictions=self._azimuthPred, targets=azimuthGT, k=5),tf.float32))

        elevationPrediction = tf.argmax(self._elevationPred, -1)
        elevationGT = tf.argmax(self._elevationGT, -1)
        elevationEquality = tf.equal(elevationPrediction, elevationGT)
        self._elevationAcc = tf.reduce_mean(tf.cast(elevationEquality, tf.float32))
        self._elevationTop5Acc = tf.reduce_mean(tf.cast(
            tf.nn.in_top_k(predictions=self._elevationPred, targets=elevationGT, k=5), tf.float32))

        in_plane_rotPrediction = tf.argmax(self._in_plane_rotPred, -1)
        in_plane_rotGT = tf.argmax(self._in_plane_rotGT, -1)
        in_plane_rotEquality = tf.equal(in_plane_rotPrediction, in_plane_rotGT)
        self._in_plane_rotAcc = tf.reduce_mean(tf.cast(in_plane_rotEquality, tf.float32))
        self._in_plane_rotTop5Acc = tf.reduce_mean(tf.cast(
            tf.nn.in_top_k(predictions=self._in_plane_rotPred, targets=in_plane_rotGT, k=5), tf.float32))

    def fit(self, batchDict):
        feed_dict = {
            self._inputImages : batchDict['inputImages'],
            self._classGT : batchDict['classList'],
            self._instGT : batchDict['instList'],
            self._azimuthGT : batchDict['azimuth'],
            self._elevationGT : batchDict['elevation'],
            self._in_plane_rotGT : batchDict['in_plane_rot'],
            self._learningRate : batchDict['learningRate']
        }
        opt, totalLoss, classAcc, instAcc, azAcc, azTop5Acc, elAcc, elTop5Acc, ipAcc, ipTop5Acc = self._sess.run(
            [self._optimizer, self._loss,
             self._classAcc, self._instAcc,
             self._azimuthAcc, self._azimuthTop5Acc,
             self._elevationAcc, self._elevationTop5Acc,
             self._in_plane_rotAcc, self._in_plane_rotTop5Acc
             ], feed_dict=feed_dict)
        return totalLoss, classAcc, instAcc, azAcc, azTop5Acc, elAcc, elTop5Acc, ipAcc, ipTop5Acc

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.save(self._sess, eLastPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)