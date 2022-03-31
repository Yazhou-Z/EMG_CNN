# Electromyography (EMG) based CNN for hand gesture classification.

**CNN classifier and sEMG dataset construction:** Convolutional neural network which contains one input layer, three hidden layers and one fully connected output layer was constructed for sEMG classification. Rectifier linear unit (ReLU) and signmoid function were chose to be the activation function after the hidden layers and the output layer respectively. Published [sEMG dataset][https://github.com/Yazhou-Z/EMG/tree/main/Kaggle_dataset] from Kaggle was harnessed for quantifying the performance of our preliminary reservoir computing model. The features (showed in appendix 1) of each group from the total 630 sets were extracted and labelled with their corresponding gestures (showed in appendix 2). The value of features was normalized to eliminate the random errors from the difference among groups, and the interpolation and slice were also conducted to expend the dataset. To assess the performance of the CNN classifier, the dataset was split randomly into two groups: 70% for training while the rest 30% for training, and the cross-validation was iterated three times respectively.

**Convolutional neural network (CNN) training.** As mentioned in the Method and Material part, 70%-30% random split was performed to train the model to cross-validate the performance and accuracy of our CNN classifier. After changing the hyperparameter, a final accuracy of 0.832 and a final loss of 1.112 were gained with the training of 20,000 epochs. But some further optimization is still needed, as the target accuracy is 95%. The iteration graph can be referred to figure below.

##### Accuracy:

<img src="https://user-images.githubusercontent.com/76484768/138593850-8396950d-48df-49af-ba2a-a560def1ffe6.png" alt="image" style="zoom: 33%;" />

##### Loss:

<img src="https://user-images.githubusercontent.com/76484768/138593867-d5e59a37-ad75-40fb-89c6-809eb7f75861.png" alt="image" style="zoom:33%;" />

**Appendix**

1. **The features of published sEMG data are:**

   a)  Standard deviation

   b)  Root mean square

   c)  Minimum value

   d)  Maximum value

   e)  Zero crossing

   f)  Average amplitude change

   g)  Amplitude first burst

   h)  Mean absolute value

   i)  Wave form length

   j)  Willison amplitude

2. **The labels of published sEMG data are:**

   a)  Index finger

   b)  Middle finger

   c)  Ring finger

   d)  Little finger

   e)  Thumb

   f)  Rest

   g)  Victory gesture

