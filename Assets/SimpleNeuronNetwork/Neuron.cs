using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SimpleNeuronNetwork
{
    public class Neuron
    {
        private int inputNumber;
        private float[] weights;
        private float bias;

        // a delegate for activation function
        public delegate float ActivationFunction(float input);
        private ActivationFunction activationFunction;

        // memory for inputs & outputs
        //public float[] inputs;
        public float output;
        //public bool isProcessing = false; // a marker for identifying work state of neurons

        /// <summary>
        /// Create a new Neuron
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        public Neuron(float[] weights, float bias)
        {
            inputNumber = weights.Length;
            this.weights = weights;
            this.bias = bias;

            // Default activation function as sigmoid function
            activationFunction = MathFunctions.Sigmoid;

            // pre allocated memory for inputs and outputs
            //inputs = new float[inputNumber];
        }

        public int GetInputNumber()
        {
            return inputNumber;
        }

        public void ChangeWeight(int index, float weight)
        {
            weights[index] = weight;
        }

        public void ChangeBias(float bias)
        {
            this.bias = bias;
        }

        public void SetActivationFunction(ActivationFunction newActivationFunc)
        {
            activationFunction = newActivationFunc;
        }

        /// <summary>
        /// Passing inputData forward to get an output
        ///  (output data will also be stored in Neuron.output)
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public void FeedForward(float[] inputData)
        {
            // input number have to match number of weight
            if(inputData.Length != inputNumber)
            {
                Debug.Log("input number does not match the number of weight");
                return;
            }

            float sum = 0;
            
            // perform dot product on every input and their weights
            for(int i = 0; i < inputNumber; i++)
            {
                sum += weights[i] * inputData[i];
            }

            // count in bias
            sum += bias;

            // use activation function
            sum = activationFunction(sum);

            output = sum;
            return;
        }

        /// <summary>
        /// Passing inputData forward to get an output. This version of FeedForward also output intrimSums for training purposes
        ///  (output data will also be stored in Neuron.output)
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public void FeedForward(float[] inputData, out float[] intrimSums)
        {
            // input number have to match number of weight
            if (inputData.Length != inputNumber)
            {
                Debug.Log("input number does not match the number of weight");
                intrimSums = null;
                return;
            }

            float sum = 0;
            intrimSums = new float[inputNumber];

            // perform dot product on every input and their weights
            for (int i = 0; i < inputNumber; i++)
            {
                float sumI = weights[i] * inputData[i];
                intrimSums[i] = sumI;
                sum += sumI;
            }

            // count in bias
            sum += bias;

            // use activation function
            sum = activationFunction(sum);

            output = sum;
            return;
        }
    }
    
}
