﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Synthetics
{
    class Membrans : Compartment
    {
        public int mSizePoint;

        public Membrans(List<ICompartment> compartments, Size sizeImg)
        {
            mPen = new Pen(Color.FromArgb(0, 128, 0), 1);
            mListPointWithOffset = new List<Point>();
            Create(compartments, sizeImg);
        }

        //Переименовывает имена меток в матрице изображения меток
        private void RenameRegion(int[,] labels, int ChangeLabel, int NewLabel)
        {
            for (int y = 0; y < labels.GetLength(1); ++y)
            {
                for (int x = 0; x < labels.GetLength(0); ++x)
                {
                    if (labels[y,x] == ChangeLabel)
                    {
                        labels[y,x] = NewLabel;
                    }
                }
            }
        }

        //Ищет метку в матрице изображения меток
        private bool FindName(int[,] labels, int findLabel)
        {
            for (int y = 0; y < labels.GetLength(1); ++y)
            {
                for (int x = 0; x < labels.GetLength(0); ++x)
                {
                    if (labels[y, x] == findLabel)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        //Функция посчёта регионов на изображении
        private void RegionCounter(Bitmap img, int[,] labels)
        {
            int label = 0;
            // Цикл по пикселям изображения
            //       C                 (ky, x)
            //     B A         (y, kx) ( y, x)
            // A - рассматриваемый в данный момент пиксель
            for (int y = 0; y < img.Height; ++y)
            {
                for (int x = 0; x < img.Width; ++x)
                {
                    int kx = x - 1;
                    int B = 0;
                    if (kx < 0)
                    {
                        kx = 0;
                    }
                    else
                    {
                        B = labels[y,kx];
                    }

                    int ky = y - 1;
                    int C = 0;
                    if (ky < 0)
                    {
                        ky = 0;
                    }
                    else
                    {
                        C = labels[ky,x];
                    }

                    int A = img.GetPixel(x, y).R;
                    if (A == 0)
                    {
                    }
                    else if (B == 0 & C == 0)
                    {
                        label = label + 1;
                        labels[y,x] = label;
                    }
                    else if (B != 0 & C == 0) {
                        labels[y,x] = B;
                    }
                    else if (B == 0 & C != 0) {
                        labels[y,x] = C;
                    }
                    else if (B != 0 & C != 0) {
                        if (B == C)
                        {
                            labels[y,x] = B;
                        }
                        else
                        {
                            labels[y,x] = B;
                            RenameRegion(labels, C, B);
                        }
                    }
                }
            }
        }

        // Приведение меток в непрерывную последовательность в диапозоне [0:N], где N от 0 до 255
        void ChangeNameRegion(int[,] labels)
        {
            int label = 1;

            while (label != 256)
            {
                if (FindName(labels, label))
                {
                    ++label;
                }
                else
                {
                    int minLabel = -1;
                    for (int y = 0; y < labels.GetLength(1); y++)
                    {
                        if (minLabel != -1)
                        {
                            break;
                        }
                        for (int x = 0; x < labels.GetLength(0); x++)
                            if (labels[y, x] > label)
                            {
                                minLabel = labels[y, x];
                                break;
                            }
                    }
                    if (minLabel != -1)
                    {
                        RenameRegion(labels, minLabel, label);
                        ++label;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            Console.WriteLine($"Max Label {label - 1}");
        }

        // Перевод 2D массива (или матрицу изображения меток) в Bitmap для возможности сохранить как png
        Bitmap CreateBitmap(int[,] source)
        {
            ChangeNameRegion(source);
            var bitmap = new Bitmap(source.GetLength(0), source.GetLength(1));
            for (var i = 0; i < bitmap.Height; i++)
                for (var j = 0; j < bitmap.Width; j++)
                    bitmap.SetPixel(j, i, Color.FromArgb(source[i, j], 0, 0)); // вместо Color.FromArgb можете использовать любой другой способ преобразования элемента массива в цвет

            return bitmap;
        }

        // вспомогательная функция для GetConvexHull
        private static double cross(Point O, Point A, Point B)
        {
            return (A.X - O.X) * (B.Y - O.Y) - (A.Y - O.Y) * (B.X - O.X);
        }

        // функция создания выпуклой оболочки                                                               ///(взял с форума, может быть плохой, но вроде работает)
        public static List<Point> GetConvexHull(List<Point> points)
        {
            if (points == null)
                return null;

            if (points.Count() <= 1)
                return points;

            int n = points.Count(), k = 0;
            List<Point> H = new List<Point>(new Point[2 * n]);

            points.Sort((a, b) =>
                 a.X == b.X ? a.Y.CompareTo(b.Y) : a.X.CompareTo(b.X));

            // Build lower hull
            for (int i = 0; i < n; ++i)
            {
                while (k >= 2 && cross(H[k - 2], H[k - 1], points[i]) <= 0)
                    k--;
                H[k++] = points[i];
            }

            // Build upper hull
            for (int i = n - 2, t = k + 1; i >= 0; i--)
            {
                while (k >= t && cross(H[k - 2], H[k - 1], points[i]) <= 0)
                    k--;
                H[k++] = points[i];
            }

            return H.Take(k - 1).ToList();
        }

        // Функция создания мембран
        public void Create(List<ICompartment> compartments, Size sizeImg)
        {
            // Создание маски регионов и зпаолнение её
            Bitmap checkImage = new Bitmap(sizeImg.Width, sizeImg.Height);
            Graphics g = Graphics.FromImage(checkImage);
            g.Clear(Color.Black);                                                                           ///есть прозрачный transparent
            SolidBrush brush = new SolidBrush(Color.White);
            Pen pen = new Pen(Color.White);
            foreach (ICompartment c in compartments)
            {
                if (c.GetType() != typeof(Vesicules))                                                       /// можно обобщить для всех кроме класса PSD и мембран
                {
                    c.DrawMask(g);
                }
                else
                {
                    Vesicules ves_c = (Vesicules)c;
                    List<Point> ConvexHuLLVesicules = GetConvexHull(ves_c.mListPointWithOffset);
                    pen.Width = ves_c.mSizeCycle.Width + 2;


                    g.FillClosedCurve(brush, ConvexHuLLVesicules.ToArray());
                    g.DrawClosedCurve(pen, ConvexHuLLVesicules.ToArray());

                }
            }

            // Сохранение маски
            //checkImage.Save("fileMask.png", System.Drawing.Imaging.ImageFormat.Png);

            // Создание и инициализация 0 матрицы меток (или матрицы изображения меток)
            int[,] labelImage = new int[sizeImg.Height, sizeImg.Width];

            for (int y = 0; y < sizeImg.Height; ++y)
            {
                for (int x = 0; x < sizeImg.Width; ++x)
                {
                    labelImage[y,x] = 0;
                }
            }

            // Вычисление карты различных регионов
            RegionCounter(checkImage, labelImage);
            //CreateBitmap(labelImage).Save("file.png", System.Drawing.Imaging.ImageFormat.Png);

            // Разрастание регионов
            bool repit = true;
            while (repit) // Продолжать пока есть новые пиксели
            {
                int[,] LastLabel = (int[,])labelImage.Clone();
                repit = false;
                for (int y = 0; y < sizeImg.Height; ++y)
                {
                    for (int x = 0; x < sizeImg.Width; ++x)
                    {
                        // Для каждого ненулевого пикселя проверить ближайших соседей (пустой, тот же регион или граница с другим)
                        if (LastLabel[y, x] != 0)
                        {
                            int label = LastLabel[y, x];
                            // сосед слева
                            if (x > 0)
                            {
                                int value = labelImage[y, x - 1];
                                if (value == 0)                   // Если есть неразмеченный сосед, то пометить его своей меткой
                                {
                                    labelImage[y, x - 1] = label;
                                    repit = true;                 // Если есть новый пиксель, то проверить что из него можно разростись на следующей итерации
                                }
                                else if (value != label)
                                {
                                    mPoints.Add(new Point(x, y)); // Добавление граничного пикселя
                                }
                            }

                            // сосед справа
                            if (x < sizeImg.Width - 1)
                            {
                                int value = labelImage[y, x + 1];
                                if (value == 0)
                                {
                                    labelImage[y, x + 1] = label;
                                    repit = true;
                                }
                                else if (value != label)
                                {
                                    mPoints.Add(new Point(x, y));  // Добавление граничного пикселя
                                }
                            }

                            // сосед сверху
                            if (y > 0)
                            {
                                int value = labelImage[y - 1, x];
                                if (value == 0)
                                {
                                    labelImage[y - 1, x] = label;
                                    repit = true;
                                }
                                else if (value != label)
                                {
                                    mPoints.Add(new Point(x, y));  // Добавление граничного пикселя
                                }
                            }

                            // сосед снизу
                            if (y < sizeImg.Height - 1)
                            {
                                int value = labelImage[y + 1, x];
                                if (value == 0)
                                {
                                    labelImage[y + 1, x] = label;
                                    repit = true;
                                }
                                else if (value != label)
                                {
                                    mPoints.Add(new Point(x, y));  // Добавление граничного пикселя
                                }
                            }
                        }
                    }
                }
            }

            //Сохранение карты разросшихся регионов
            //CreateBitmap(labelImage).Save("file2.png", System.Drawing.Imaging.ImageFormat.Png);

            //печать в консоль доп. информацию
            Console.WriteLine($"count point = {mPoints.Count}");
        }

        public override void Draw(Graphics g)
        {
            // Попиксельная печаль граничных пикселей (у каждого будет по 2 штуки с одного и с другого региона)
            foreach (Point point in mPoints)
            {
                g.DrawLine(mPen, point.X, point.Y, point.X + 1 , point.Y + 1);
            }
        }

        SolidBrush brush = new SolidBrush(Color.Red);
        private Color mPenColor = Color.Red;
        protected override void setMaskParam()
        {
            mPenColor = mPen.Color;
            brush = new SolidBrush(Color.White);
            mPen.Color = Color.White;
        }

        protected override void setDrawParam()
        {
            brush = new SolidBrush(Color.Red);
            mPen.Color = mPenColor;
        }

    }
}
