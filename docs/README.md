|arxiv|   |gitter| |mendeley| |contributions-welcome|  |researchgate|  |GitHublicense|  |contributors| |twitter|

Referenced paper : `HHAR-net: Hierarchical Human Activity Recognition using Neural Networks <https://arxiv.org/abs/2010.16052>`__


HHAR-net: Hierarchical Human Activity Recognition using Neural Networks
=======================================================================

Activity recognition using built-in sensors in smart and wearable devices provides great opportunities to understand and detect human behavior in the wild and gives a more holistic view of individuals' health and well being. Numerous computational methods have been applied to sensor streams to recognize different daily activities. However, most methods are unable to capture different layers of activities concealed in human behavior. Also, the performance of the models starts to decrease with increasing the number of activities. This research aims at building a hierarchical classification with Neural Networks to recognize human activities based on different levels of abstraction. We evaluate our model on the Extrasensory dataset; a dataset collected in the wild and containing data from smartphones and smartwatches. We use a two-level hierarchy with a total of six mutually exclusive labels namely, "lying down", "sitting", "standing in place", "walking", "running", and "bicycling" divided into "stationary" and "non-stationary". 


.. image:: docs/pic/HHAR-net.PNG



Hierarchical Classification
===========================
Hierarchical Classification algorithms employ stacks of machine learning architectures to provide specialized understanding at each level of the data hierarchy which have been used in many domains such as text and document classification, medical image classification, web content and sensor data. Human activity recognition is a multi-class (and in some cases multi-label) classification task. Examples of activities include sitting, walking, eating, watching TV, and bathing.



.. image:: docs/pic/HHAR.PNG


-----------------------------------------
Deep Neural Networks
-----------------------------------------

Deep Neural Networks architectures are designed to learn through multiple connection of layers where each single layer only receives connection from previous and provides connections only to the next layer in hidden part. The input is a connection of feature space with first hidden layer. For Deep Neural Networks (DNN), input layer is from Sensors as shown in standard DNN in Figure. The output layer houses neurons equal to the number of classes for multi-class classification and only one neuron for binary classification. Our implementation of Deep Neural Network (DNN) is basically a discriminatively trained model that uses standard back-propagation algorithm and sigmoid or ReLU as activation functions. The output layer for multi-class classification should use Softmax.


.. image:: docs/pic/DNN.png

--------
Dataset
--------


The Extrasensory dataset that is used in this work is a publicly available dataset collected by `Vaizman <https://arxiv.org/pdf/1609.06354.pdf>`__ at the University of California San Diego. This data was collected in the wild using smartphones and wearable devices. The users were asked to provide the ground truth label for their activities. 
The dataset contains over 300k samples (one sample per minute) and 50 activity and context labels collected from 60 individuals. Each label has its binary column indicating whether that label was relevant or not. Data was collected from accelerometer, gyroscope, magnetometer, watch compass, location, audio, and phone state sensors. Both  featurized and raw data are provided in the dataset. Featurized data has a total of 225 features extracted from six main sensors. We used the featurized data in this work.


-----------------------------------------
Evaluation and Results
-----------------------------------------

.. image:: docs/pic/Results_CM.PNG



==========
Citations:
==========

----

.. code::

    @inproceedings{HHAR2020,
        title={HHAR-net: Hierarchical Human Activity Recognition using Neural Networks},
        author={Fazli, Mehrdad and Kowsari, Kamran and Gharavi, Erfaneh and Barnes, Laura and Doryab, Afsaneh},
        booktitle={Intelligent Human Computer Interaction: 12th International Conference, IHCI 2020, Daegu, South Korea, December 24--26, 2020, Proceedings},
        organization={Springer Nature}
    }


.. |contributors| image:: https://img.shields.io/github/contributors/mehrdadfazli/HHAR-Net.svg
      :target: https://github.com/mehrdadfazli/HHAR-Net/graphs/contributors 
      
.. |contributions-welcome| image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/mehrdadfazli/HHAR-Net/pulls

.. |researchgate| image:: https://img.shields.io/badge/ResearchGate-HHAR_net-blue.svg?style=flat
   :target: https://www.researchgate.net/publication/344934245_HHAR-net_Hierarchical_Human_Activity_Recognition_using_Neural_Networks
   
.. |GitHublicense| image:: https://img.shields.io/badge/licence-AGPL-blue.svg
   :target: ./LICENSE
.. |arxiv| image:: https://img.shields.io/badge/arXiv-2010.16052-red.svg
    :target: https://arxiv.org/abs/2010.16052
.. |twitter| image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social
    :target: https://twitter.com/intent/tweet?text=HHAR-net:%20Hierarchical%20Human%20Activity%20Recognition%20using%20Neural%20Networks%0aGitHub:&url=https://github.com/mehrdadfazli/HHAR-Net&hashtags=DeepLearning,ActivityRecognition,MachineLearning,deep_neural_networks,DataScience

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/HHAR-net/community

.. |mendeley| image:: https://img.shields.io/badge/Mendeley-Add%20to%20Library-critical.svg
    :target: https://www.mendeley.com/import/?url=https://arxiv.org/abs/2010.16052
