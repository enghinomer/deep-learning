import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Class representing a neural network. It uses the Cross-entropy cost function.
 * L2 regularization is applied
 * */

public class NetworkCustom {
    private int layers;
    private int[] layerSizes;
    private List<INDArray> biases = new ArrayList<>();
    private List<INDArray> weights = new ArrayList<>();

    /**
     * The constructor takes as input and array of integers
     * representing the number of nodes in each layer
     * */
    public NetworkCustom(int[] layerSizes) {
        this.layerSizes = Arrays.copyOf(layerSizes, layerSizes.length);
        this.layers = layerSizes.length;

        //initialise biases
        for (int i = 1; i<layerSizes.length; i++) {
            biases.add(Nd4j.randn(layerSizes[i], 1));
        }

        //initialise weights
        for (int i = 1; i<layerSizes.length; i++) {
            weights.add(Nd4j.randn(layerSizes[i], layerSizes[i-1]).div(Math.sqrt(layerSizes[i-1])));
        }
    }

    /**
     * Given an input computes the output of the network
     * @param input the input example for the network
     * @return output of the network
     * */
    public INDArray feedForward(INDArray input) {
        INDArray a = input;
        for (int i = 0; i < layers-1; i++) {
            a = sigmoid(weights.get(i).mmul(a).add(biases.get(i)));
        }
        return a;
    }

    /**
     * Performs mini-batch gradient descent to train the network. If test data is provided
     * it will print the performance of the network at each epoch
     * @param trainingData data to train the network
     * @param epochs number of epochs used to train the network
     * @param batchSize the size of batch
     * @param eta the learning rate
     * @param lambda the regularization parameter
     * @param testData data to test the network
     * */
    public void SGD(DataSet trainingData, int epochs, int batchSize, double eta, double lambda, DataSet testData) {
        int testSize=0;
        if (testData != null) {
            testSize = testData.numExamples();
        }
        int trainingSize = trainingData.numExamples();
        for (int i=0; i<epochs; i++) {
            trainingData.shuffle();
            for(int j=0; j<trainingSize; j+=batchSize) {
                DataSet miniBatch = trainingData.getRange(j, j+batchSize<trainingSize ? j+batchSize : trainingSize-1);
                this.updateMiniBatch(miniBatch, eta, lambda, trainingSize);
            }
            if (testData != null) {
                System.out.printf("Epoch %s: %d / %d ", i, this.evaluate(testData), testSize);
                System.out.println();
            }
        }
    }

    /**
     * Updates the weights un biases of the network using backpropagation for a single mini-batch
     * @param miniBatch the mini batch used to train the network
     * @param eta the learning rate
     * @param lambda the regularization parameter
     * @param trainingSize the size of the training data
     * */
    public void updateMiniBatch(DataSet miniBatch, double eta, double lambda, int trainingSize) {
        INDArray [] gradientBatchB = new INDArray[layers];
        INDArray [] gradientBatchW = new INDArray[layers];
        for (int i=0; i < this.biases.size(); i++) {
            gradientBatchB[i+1] = Nd4j.zeros(this.biases.get(i).shape());
        }
        for (int i=0; i < this.weights.size(); i++) {
            gradientBatchW[i+1] = Nd4j.zeros(this.weights.get(i).shape());
        }
        List<INDArray[]> result;
        for(DataSet batch : miniBatch) {
            result = this.backpropagation(batch.getFeatures(), batch.getLabels());
            for(int i=1; i<layers; i++) {
                gradientBatchB[i] = gradientBatchB[i].add(result.get(0)[i]);
                gradientBatchW[i] = gradientBatchW[i].add(result.get(1)[i]);
            }
        }
        for (int i=0; i<this.biases.size(); i++) {
            INDArray b = this.biases.get(i).sub(gradientBatchB[i+1].mul(eta/miniBatch.numExamples()));
            this.biases.set(i, b);
            INDArray w = (this.weights.get(i).mul(1-eta*(lambda/trainingSize))).sub(gradientBatchW[i+1].mul(eta/miniBatch.numExamples()));
            this.weights.set(i, w);
        }
    }

    /**
     * Method used to compute the partial derivatives of cost C with respect to weights and biases
     * @param x a single feature data
     * @param y the corresponding label
     * @return List containing at index 0 biases gradient and at index 1 weights gradient
     * */
    public List<INDArray[]> backpropagation(INDArray x, INDArray y) {
        x.transposei();
        y.transposei();
        List<INDArray[]> result = new ArrayList<>();
        INDArray [] gradientB = new INDArray[layers];
        INDArray [] gradientW = new INDArray[layers];
        result.add(gradientB);
        result.add(gradientW);
        for (int i=0; i < this.biases.size(); i++) {
            gradientB[i+1] = Nd4j.zeros(this.biases.get(i).shape());
        }
        for (int i=0; i < this.weights.size(); i++) {
            gradientW[i+1] = Nd4j.zeros(this.weights.get(i).shape());
        }
        //feedforward
        INDArray activation = x;
        INDArray [] activations = new INDArray[layers];
        INDArray [] zs = new INDArray[layers];
        activations[0] = x;
        INDArray z;
        for (int i=1; i<layers; i++) {
            z = this.weights.get(i-1).mmul(activation).add(this.biases.get(i-1));
            zs[i] = z;
            activation = sigmoid(z);
            activations[i] = activation;
        }

        //back pass
        INDArray sp;
        INDArray delta = costDerivative(activations[layers-1], y);
        gradientB[layers - 1] = delta;
        gradientW[layers - 1] = delta.mul(activations[layers-2].transpose());
        for (int i=2; i<layers; i++) {
            z = zs[layers-i];
            sp = sigmoidPrime(z);
            delta = (this.weights.get(layers - i).transpose().mmul(delta)).mul(sp);
            gradientB[layers - i] = delta;
            gradientW[layers - i] = delta.mmul(activations[layers - i - 1].transpose());
        }
        return result;
    }

    /**
     * @param testData the test Data
     * @return number of correctly classified inputs
     * */
    public int evaluate(DataSet testData) {
        int sum = 0;
        for (DataSet input : testData) {
            INDArray argMax = this.feedForward(input.getFeatures().transpose()).argMax(0);
            if (argMax.getInt(0) == input.getLabels().argMax(1).getInt(0)) {
                sum++;
            }
        }
        return sum;
    }

    /**
    *@param outputActivations the output of th network
    *@param y the label of the input example x
    *@return the partial derivatives \partial C(x) / \partial a for the output activation
    */
    public INDArray costDerivative(INDArray outputActivations, INDArray y) {
        return outputActivations.sub(y);
    }

    /**
     * @param z input for which to apply the sigmoid function
     * @return result of sigmoid function applied to each entry of the input
     * */
    public INDArray sigmoid(INDArray z) {
        return Transforms.sigmoid(z);
    }

    /**
     * @param z input for which to apply the first order derivative of sigmoid function
     * @return result of irst order derivative of sigmoid function applied to each entry of the input
     * */
    public INDArray sigmoidPrime(INDArray z) {
        return Transforms.sigmoid(z).mul(Nd4j.ones(z.shape()).sub(Transforms.sigmoid(z)));
    }

    public static void main(String[] args) throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(60000, true, 12);
        DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 12);
        int [] sizes = {784, 30, 10};
        NetworkCustom network = new NetworkCustom(sizes);
        network.SGD(mnistTrain.next(), 30, 10, 0.5, 5.0, mnistTest.next());
    }
}
