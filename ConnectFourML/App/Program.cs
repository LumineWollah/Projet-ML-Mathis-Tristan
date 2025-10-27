using System;
using System.IO;
using ConnectFour.Core;

namespace App
{
    class Program
    {
        static void Main()
        {
            var baseDir = AppContext.BaseDirectory;
            var dataPath = Path.Combine(baseDir, "dataset.csv");
            var modelPath = Path.Combine(baseDir, "model.zip");

            // création du dataset
            if (!File.Exists(dataPath))
            {
                Console.WriteLine("Génération du dataset...");
                DatasetGenerator.GenerateCsv(dataPath, games: 100000);
            }

            // entrainement
            Trainer.Train(dataPath, modelPath);

            // 3) Démo d'inférence sur TON jeu
            var p1 = new Player("Humain", 'X');
            var p2 = new Player("IA", 'O');
            var game = new Game(p1, p2);
            using var infer = new Inference(modelPath);

            while (true)
            {
                // affichage simple
                PrintBoard(game.Board);

                // check fin
                bool full = game.Board.LegalMoves().GetEnumerator().MoveNext() == false;
                if (full) { Console.WriteLine("Match nul (plateau plein)."); break; }

                if (game.CurrentPlayer == p1)
                {
                    Console.Write("Colonne (0-6) : ");
                    if (!int.TryParse(Console.ReadLine(), out int col)) continue;
                    if (!game.Board.PlaceDisc(col, p1.Disc, out int r)) { Console.WriteLine("Colonne pleine."); continue; }
                    if (Rules.CheckWin(game.Board, r, col)) { PrintBoard(game.Board); Console.WriteLine("Victoire HUMAIN"); break; }
                    game.SwitchTurn();
                }
                else
                {
                    int col = infer.PredictColumn(game.Board, game.CurrentPlayer.Disc.Symbol);
                    game.Board.PlaceDisc(col, p2.Disc, out int r);
                    Console.WriteLine($"IA joue colonne {col}");
                    if (Rules.CheckWin(game.Board, r, col)) { PrintBoard(game.Board); Console.WriteLine("Victoire IA"); break; }
                    game.SwitchTurn();
                }
            }
        }

        static void PrintBoard(Board b)
        {
            for (int r = 0; r < Board.Rows; r++)
            {
                for (int c = 0; c < Board.Cols; c++)
                    Console.Write(b.GetCell(r, c)?.Symbol switch { null => '.', char ch => ch });
                Console.WriteLine();
            }
            Console.WriteLine("0123456");
        }
    }
}
