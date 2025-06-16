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

        public Neuron(float[] weights, float bias)
        {
            inputNumber = weights.Length;
            this.weights = weights;
            this.bias = bias;

            // Default activation function as sigmoid function
            activationFunction = x =>
            {
                return 1f / (1f + Mathf.Exp(-x));
            };
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
        /// Passing inputs forward to get an output
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public float FeedForward(float[] inputs)
        {
            // input number have to match number of weight
            if(inputs.Length != inputNumber)
            {
                Debug.Log("input number does not match the number of weight");
                return -1f;
            }

            float sum = 0;
            
            // perform dot product on every input and their weights
            for(int i = 0; i < inputNumber; i++)
            {
                sum += weights[i] * inputs[i];
            }

            // count in bias
            sum += bias;

            return sum;
        }


    }
    
}
