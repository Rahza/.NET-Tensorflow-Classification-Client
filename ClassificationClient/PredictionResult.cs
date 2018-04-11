using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TF_NN_ClassificationClient
{
    public class PredictionResult
    {
        public int PredictedClass { get; set; }
        public float[] Scores { get; set; }

        public PredictionResult()
        {
        }
    }
}
