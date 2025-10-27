using Microsoft.ML.Data;

namespace MLCommon
{
    public class GameExample
    {
        [LoadColumn(0, 41)]
        [VectorType(42)]
        public float[] Features { get; set; } = new float[42];

        [LoadColumn(42)]
        public float Label { get; set; }
    }

    public class GamePrediction
    {
        [ColumnName("PredictedLabel")] public float PredictedLabel { get; set; }
        public float[] Score { get; set; } = System.Array.Empty<float>();
    }
}
