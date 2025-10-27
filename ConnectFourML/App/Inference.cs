using System;
using System.Linq;
using Microsoft.ML;
using MLCommon;
using ConnectFour.Core;

namespace App
{
    public sealed class Inference : IDisposable
    {
        private readonly MLContext _ml = new();
        private readonly PredictionEngine<GameExample, GamePrediction> _engine;

        public Inference(string modelPath)
        {
            var model = _ml.Model.Load(modelPath, out _);
            _engine = _ml.Model.CreatePredictionEngine<GameExample, GamePrediction>(model);
        }

        public int PredictColumn(Board b, char currentSymbol)
        {
            var ex = new GameExample { Features = Encoders.Encode42(b, currentSymbol) };
            var pred = _engine.Predict(ex);

            var scores = (float[])pred.Score.Clone();
            var legal = b.LegalMoves().ToHashSet();
            for (int c = 0; c < Board.Cols; c++)
                if (!legal.Contains(c)) scores[c] = float.NegativeInfinity;

            int best = 0; float bestV = scores[0];
            for (int c = 1; c < Board.Cols; c++)
                if (scores[c] > bestV) { bestV = scores[c]; best = c; }
            return best;
        }

        public void Dispose() => _engine.Dispose();
    }
}
