using UnityEngine;

namespace SimpleNeuronNetwork
{
    public class NeuralNetwork
    {
        private float[] inputs;
        private Neuron[][] neurons;

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
        }

        public void SetInputs(float[] inputs)
        {
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


    }
}
