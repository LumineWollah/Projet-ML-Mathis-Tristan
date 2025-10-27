using System;
using System.IO;
using System.Linq;
using System.Text;
using ConnectFour.Core;
using MLCommon;

namespace App
{
    public static class DatasetGenerator
    {
        public static void GenerateCsv(string path, int games = 4000, int maxMoves = Board.Rows * Board.Cols)
        {
            var rnd = new Random(42);
            var sb = new StringBuilder();
            // header
            for (int i = 0; i < 42; i++) sb.Append($"f{i+1},");
            sb.AppendLine("Label");

            for (int g = 0; g < games; g++)
            {
                var p1 = new Player("P1", 'X');
                var p2 = new Player("P2", 'O');
                var game = new Game(p1, p2); // CurrentPlayer = p1
                var board = game.Board;

                int moves = 0;
                bool ended = false;

                while (moves < maxMoves && !ended)
                {
                    var legal = board.LegalMoves().ToList();
                    if (legal.Count == 0) break;

                    // petit biais: favoriser colonne du centre 3 si possible
                    int move;
                    if (legal.Contains(3) && rnd.NextDouble() < 0.35) move = 3;
                    else move = legal[rnd.Next(legal.Count)];

                    // enregistrer (état avant coup -> colonne choisie)
                    var features = Encoders.Encode42(board, game.CurrentPlayer.Disc.Symbol);
                    for (int i = 0; i < 42; i++) sb.Append(features[i].ToString(System.Globalization.CultureInfo.InvariantCulture)).Append(',');
                    sb.AppendLine(move.ToString());

                    // jouer
                    if (!board.PlaceDisc(move, game.CurrentPlayer.Disc, out int rowPlaced))
                        break; // devrait pas arriver si legal

                    moves++;

                    // victoire ?
                    if (Rules.CheckWin(board, rowPlaced, move))
                    {
                        ended = true;
                        break;
                    }

                    // tour suivant
                    game.SwitchTurn();
                }
            }

            File.WriteAllText(path, sb.ToString());
        }
    }
}
