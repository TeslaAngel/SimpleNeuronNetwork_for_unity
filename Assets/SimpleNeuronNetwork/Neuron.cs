using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SimpleNeuronNetwork
{
    public class Neuron
    {
        private uint inputNumber;
        private float[] weights;
        private float bias;

        // a delegate for activation function
        public delegate float ActivationFunction(float input);
        private ActivationFunction activationFunction;

        public Neuron(uint inputNumber, float[] weights, float bias)
        {
            this.inputNumber = inputNumber;
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
    }
    
}
