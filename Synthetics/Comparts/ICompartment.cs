﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Synthetics
{
    interface ICompartment
    { 
        void Draw(Graphics g);
        void DrawMask(Graphics g);
        void NewPosition(int x, int y);
    }
}