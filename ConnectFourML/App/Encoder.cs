using ConnectFour.Core;

namespace App
{
    public static class Encoders
    {
        public static float[] Encode42(Board b, char currentSymbol)
        {
            var v = new float[Board.Rows * Board.Cols];
            for (int r = 0; r < Board.Rows; r++)
            for (int c = 0; c < Board.Cols; c++)
            {
                var cell = b.GetCell(r, c);
                if (cell == null) continue;
                v[r * Board.Cols + c] = (cell.Symbol == currentSymbol) ? 1f : -1f;
            }
            return v;
        }
    }
}
