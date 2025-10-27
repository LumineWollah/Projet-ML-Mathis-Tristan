using Microsoft.ML;
using MLCommon;

namespace App
{
    public static class Trainer
    {
        public static void Train(string csvPath, string modelPath)
        {
            var ml = new MLContext(seed: 123);

            var data = ml.Data.LoadFromTextFile<GameExample>(csvPath, hasHeader: true, separatorChar: ',');
            var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipe =
                ml.Transforms.Concatenate("Features", nameof(GameExample.Features))
                .Append(ml.Transforms.Conversion.MapValueToKey("Label"))
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
                .AppendCacheCheckpoint(ml);

            var model = pipe.Fit(split.TrainSet);

            var preds = model.Transform(split.TestSet);
            var metrics = ml.MulticlassClassification.Evaluate(preds, labelColumnName: "Label");
            System.Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2} | MicroAccuracy: {metrics.MicroAccuracy:P2}");

            ml.Model.Save(model, split.TrainSet.Schema, modelPath);
            System.Console.WriteLine($"Modèle sauvegardé → {modelPath}");
        }
    }
}
