using System;
using System.Threading;
using UnityEngine;
using static UnityEditor.Experimental.GraphView.GraphView;

namespace SimpleNeuronNetwork
{

    /// <summary>
    /// A fully-connected neural network
    /// </summary>
    public class NeuralNetwork
    {
        private float[] inputs;
        private Neuron[][] neurons;

        // a delegate for loss function
        public delegate float LossFunction(float[] predictions, float[] targets);
        private LossFunction lossFunction;

        /// <summary>
        /// create a neural network with predefined neurons
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="hiddenLayerNeurons"></param>
        /// <param name="outputLayerNeuron"></param>
        public NeuralNetwork(int inputNumber, Neuron[][] neurons)
        {
            inputs = new float[inputNumber];
            this.neurons = neurons;

            // default loss function as mean squared error (MSE) loss
            lossFunction = MathFunctions.MeanSquaredError;
        }

        public void SetInputs(float[] inputs)
        {
            if(inputs == null || inputs.Length != this.inputs.Length)
            {
                Debug.Log("This input is invalid");
                return;
            }

            this.inputs = inputs;
        }

        public void ReplaceNeuron(int layer, int index, Neuron newNeuron)
        {
            if (layer >= neurons.Length || layer >= neurons[layer].Length || layer < 0 || index < 0)
            {
                Debug.Log("Can't replace neuron: this neuron does not exist");
                return;
            }

            neurons[layer][index] = newNeuron;
        }

        public void SetLossFunction(LossFunction newLossFunction)
        {
            lossFunction = newLossFunction;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public float[] FeedForward()
        {
            float[] outputs = null;

            // run neurons layer by layer
            for (int l = 0; l < neurons.Length; l++)
            {
                Thread[] neuronThreads = new Thread[neurons[l].Length];

                // run neurons in threads
                for(int i = 0; i < neurons[l].Length; i++)
                {
                    // first layer accepts inputs directly
                    if (l == 0)
                    {
                        neuronThreads[i] = new Thread(() => neurons[l][i].FeedForward(inputs));
                        neuronThreads[i].Start();
                    }
                    else
                    {
                        neuronThreads[i] = new Thread(() => neurons[l][i].FeedForward(outputs));
                        neuronThreads[i].Start();
                    }
                }

                // wait for all threads to finish
                for (int t = 0; t < neuronThreads.Length; t++)
                {
                    neuronThreads[t].Join();
                }

                // record output data in memory for next layer
                outputs = new float[neurons[l].Length];
                for (int i = 0; i < neurons[l].Length; i++)
                {
                    outputs[i] = neurons[l][i].output;
                }

                // if we have reached the last layer, return the output
                if(l ==  neurons.Length - 1)
                {
                    return outputs;
                }
            }

            Debug.Log("Feedforward completed without result");
            return null;
        }

        /// <summary>
        /// Still FeedForward, but return intrimNeuronSums for training purposes
        /// </summary>
        /// <param name="intrimNeuronSums"></param>
        /// <returns></returns>
        public float[] FeedForward(out float[][][] intrimNeuronSums)
        {
            float[] outputs = null;
            float[][][] neuronSums = new float[neurons.Length][][];

            // run neurons layer by layer
            for (int l = 0; l < neurons.Length; l++)
            {
                Thread[] neuronThreads = new Thread[neurons[l].Length];
                neuronSums[l] = new float[neurons[l].Length][];

                // run neurons in threads
                for (int i = 0; i < neurons[l].Length; i++)
                {
                    // first layer accepts inputs directly
                    if (l == 0)
                    {
                        neuronThreads[i] = new Thread(() => {
                            neuronSums[l][i] = new float[neurons[l][i].GetInputNumber()];
                            neurons[l][i].FeedForward(inputs, out neuronSums[l][i]);
                        });
                        neuronThreads[i].Start();
                    }
                    else
                    {
                        neuronThreads[i] = new Thread(() => {
                            neuronSums[l][i] = new float[neurons[l][i].GetInputNumber()];
                            neurons[l][i].FeedForward(outputs, out neuronSums[l][i]);
                        });
                        neuronThreads[i].Start();
                    }
                }

                // wait for all threads to finish
                for (int t = 0; t < neuronThreads.Length; t++)
                {
                    neuronThreads[t].Join();
                }

                // record output data in memory for next layer
                outputs = new float[neurons[l].Length];
                for (int i = 0; i < neurons[l].Length; i++)
                {
                    outputs[i] = neurons[l][i].output;
                }

                // if we have reached the last layer, return the output
                if (l == neurons.Length - 1)
                {
                    intrimNeuronSums = neuronSums;
                    return outputs;
                }
            }

            Debug.Log("Feedforward completed without result");
            intrimNeuronSums = neuronSums;
            return null;
        }

        public void Train(float[][] data, float[][] allYTrues, int epochs = 10, float learnRate = 0.1f)
        {
            // one epoch = one iteration over the whole data set
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // iterate every section of data
                for (int i = 0; i < data.Length; i++)
                {
                    // initialize data
                    float[] x = data[i];
                    float yTrue = allYTrues[i];
                    float[][][] intrimSums = null;

                    // run a FeedForward and get the intrim data
                    float[] o1 = FeedForward(out intrimSums);

                    // predicted value
                    float[] yPred = o1;

                    // get dL/dYpred


                    float dL_dYpred = -2 * (yTrue - yPred);

                    float dYpred_dW5 = h1 * DerivSigmoid(sumO1);
                    float dYpred_dW6 = h2 * DerivSigmoid(sumO1);
                    float dYpred_dB3 = DerivSigmoid(sumO1);

                    float dYpred_dH1 = w5 * DerivSigmoid(sumO1);
                    float dYpred_dH2 = w6 * DerivSigmoid(sumO1);

                    float dH1_dW1 = x[0] * DerivSigmoid(sumH1);
                    float dH1_dW2 = x[1] * DerivSigmoid(sumH1);
                    float dH1_dB1 = DerivSigmoid(sumH1);

                    float dH2_dW3 = x[0] * DerivSigmoid(sumH2);
                    float dH2_dW4 = x[1] * DerivSigmoid(sumH2);
                    float dH2_dB2 = DerivSigmoid(sumH2);

                    // Update weights and biases
                    w1 -= learnRate * dL_dYpred * dYpred_dH1 * dH1_dW1;
                    w2 -= learnRate * dL_dYpred * dYpred_dH1 * dH1_dW2;
                    b1 -= learnRate * dL_dYpred * dYpred_dH1 * dH1_dB1;

                    w3 -= learnRate * dL_dYpred * dYpred_dH2 * dH2_dW3;
                    w4 -= learnRate * dL_dYpred * dYpred_dH2 * dH2_dW4;
                    b2 -= learnRate * dL_dYpred * dYpred_dH2 * dH2_dB2;

                    w5 -= learnRate * dL_dYpred * dYpred_dW5;
                    w6 -= learnRate * dL_dYpred * dYpred_dW6;
                    b3 -= learnRate * dL_dYpred * dYpred_dB3;
                }

                // Print loss every 10 epochs
                if (epoch % 10 == 0)
                {
                    float[] yPreds = new float[data.Length];
                    for (int i = 0; i < data.Length; i++)
                        yPreds[i] = Feedforward(data[i]);
                    float loss = MSELoss(allYTrues, yPreds);
                    Debug.Log($"Epoch {epoch} loss: {loss:F3}");
                }
            }
        }
    }
}
