using ConnectFour.Core;
using System.Collections.Generic;

namespace App
{
    public static class Extensions
    {
        public static IEnumerable<int> LegalMoves(this Board b)
        {
            for (int c = 0; c < Board.Cols; c++)
                if (b.GetCell(0, c) == null) // si haut vide, on peut jouer dans la colonne
                    yield return c;
        }
    }
}
