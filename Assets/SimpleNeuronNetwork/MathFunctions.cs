using System;
using System.Threading;
using UnityEngine;
using static UnityEditor.Experimental.GraphView.GraphView;

namespace SimpleNeuronNetwork
{

    /// <summary>
    /// Simple math functions used by the neural network sample
    /// </summary>
    public static class MathFunctions
    {
        /// <summary>
        /// Sigmoid function: S(x) = 1 / (1 + e^-x)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static float Sigmoid(float x)
        {
            return 1f / (1f + Mathf.Exp(-x));
        }

        /// <summary>
        /// Partial derivative of the Sigmoid Function
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static float DerivSigmoid(float x)
        {
            float fx = Sigmoid(x);
            return fx * (1 - fx);
        }

        /// <summary>
        /// Calculates the Mean Squared Error (MSE) between two float arrays.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual target values.</param>
        /// <returns>Mean squared error as a float.</returns>
        public static float MeanSquaredError(float[] predictions, float[] targets)
        {
            if (predictions.Length != targets.Length)
            {
                Debug.LogError("Predictions and targets must be the same length.");
                return -1f;
            }

            float sum = 0f;
            for (int i = 0; i < predictions.Length; i++)
            {
                float diff = predictions[i] - targets[i];
                sum += diff * diff;
            }

            return sum / predictions.Length;
        }

    }
}
