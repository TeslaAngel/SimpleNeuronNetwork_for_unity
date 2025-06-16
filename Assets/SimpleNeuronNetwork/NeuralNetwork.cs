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
                    // first neurons accepts inputs directly
                    if(l == 0)
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
    }
}
