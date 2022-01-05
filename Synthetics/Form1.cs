﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Synthetics
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            int color = random.Next(182, 200);                           // выбор цвета фона
            backgroundСolor = Color.FromArgb(color, color, color);
            currImageViewType = TypeView.layer;
            currCreateType = TypeOrganelle.axon;
            imageSize = new Size(512, 512);
            currView = new Bitmap(imageSize.Width, imageSize.Height);
        }

        /// <summary>
        /// Preview organelles image
        /// </summary>
        Bitmap imagePrew;  
          
        /// <summary>
        /// Last created object which doesn't added to result image
        /// </summary>                                               
        ICompartment LastRemember;    
        
        /// <summary>
        /// Element list
        /// </summary>                                    
        List<ICompartment> compartments = new List<ICompartment>();  ///  добавить в форму для отображения и изменения параметров 

        enum TypeView
        {
            layer,
            maskAxon,
            maskMitoxondrion,
            maskVesicules,
            maskPSD,
            maskMembranes
        }

        enum TypeOrganelle                                              
        {
            axon,
            mitoxondrion,
            PSD,
            membranes,
            vesicles
        }

        /// <summary>
        /// Current view type (image or masks)
        /// </summary>
        TypeView currImageViewType;

        Bitmap currView;

        /// <summary>
        /// The current object being created type
        /// </summary>                                            
        TypeOrganelle currCreateType;  
                                        
        Color backgroundСolor;                                       /// добавить в форму для изменения пользователем
        Size imageSize;                                              /// добавить в форму для изменения пользователем через функцию с изменением image

        Random random = new Random();

        public static bool CompareColorR(Color a, Color b)
        {
            return a.R == b.R;
        }
        private bool CheckOverlapNewElement(Bitmap all_mask, Bitmap check_mask, int width, int height)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    if (!CompareColorR(all_mask.GetPixel(x, y), Color.Black) && !CompareColorR(check_mask.GetPixel(x, y), Color.Black))
                    {
                        return true;
                    }
                }
            }

            return false;
        }
        private void AddNewElementWithoutOverlap(int width, int height)
        {
            Bitmap checkImage = new Bitmap(width, height);
            Bitmap checkNewImage = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(checkImage);
            g.Clear(Color.Black);
            foreach (ICompartment c in compartments)
            {
                c.DrawMask(g);
            }
            Graphics g2 = Graphics.FromImage(checkNewImage);

            int counter = 0;
            int max_iter = 50;
            do
            {
                g2.Clear(Color.Black);
                LastRemember.NewPosition(random.Next(0, width), random.Next(0, height));
                LastRemember.DrawMask(g2);
                ++counter;
            } while (CheckOverlapNewElement(checkImage, checkNewImage, width, height) && counter < max_iter);

            if (counter == max_iter)
            {
                Console.WriteLine("Can't add unique position new element");
            }
        }

        private void addNewElementIntoImage(int width, int height)
        {
            // добавление нового изображения если есть
            if (LastRemember != null)
            {
                AddNewElementWithoutOverlap(width, height);
                compartments.Add(LastRemember);
                LastRemember = null;

                //чистка памяти
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        private void Draw()                                          // Основная функция для отрисовки основного поля
        {
            try
            {
                Type TypeMask = typeof(ICompartment);

                switch (currImageViewType)
                {
                    case TypeView.layer:
                        break;
                    case TypeView.maskAxon:
                        TypeMask = typeof(Acson);
                        break;
                    case TypeView.maskMitoxondrion:
                        TypeMask = typeof(Mitohondrion);
                        break;
                    case TypeView.maskVesicules:
                        TypeMask = typeof(Vesicules);
                        break;
                    case TypeView.maskPSD:
                        TypeMask = typeof(PSD);
                        break;
                    case TypeView.maskMembranes:
                        TypeMask = typeof(Membrans);
                        break;

                    default:
                        throw new Exception("Неизвестный тип отображения в Draw или ошибка в масках");
                }

                Color background = Color.Black; // выбор цвета фона в зависимости маска или изображение
                if (currImageViewType == TypeView.layer)
                {
                    background = backgroundСolor;
                }

                Graphics g = Graphics.FromImage(currView);
                // чистка изображения
                g.Clear(background);

                // рисование в зависимости от типа (изображение или маска)
                foreach (ICompartment c in compartments)
                {
                    if (currImageViewType == TypeView.layer)
                    {
                        c.Draw(g);
                    }
                    else
                    {
                        if (c.GetType() == TypeMask)
                        {
                            c.DrawMask(g);
                        }
                    }
                }

                pictureGeneralBox.Image = currView;
                pictureGeneralBox.Refresh();

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private void updateViewComponentList()
        {
            listElements.BeginUpdate();
            listElements.Items.Clear();
            foreach (ICompartment c in compartments)
            {
                listElements.Items.Add(c.GetType().Name.ToString());
            }
            listElements.EndUpdate();
        }

        private void generate_Click(object sender, EventArgs e)                                         /// Возможно будет достаточно сделать только функию добавления и рисовки этого объекта и не вызать перерисовку draw
        {
            // добавление нового изображения если есть
            addNewElementIntoImage(imageSize.Width, imageSize.Height);
            Draw();
            updateViewComponentList();
        }

        private void create_organel_Click(object sender, EventArgs e)                                   // Функция создания объектов по параметрам /// не доделана
        {
            try
            {
                if (imagePrew == null)
                {
                    imagePrew = new Bitmap(256, 256);
                }

                Graphics g = Graphics.FromImage(imagePrew);
                g.Clear(Color.White);

                if (LastRemember != null) // перерисовка, чистка памяти
                {
                    LastRemember = null;
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }

                switch (currCreateType) /// не все типы ///////////////////
                {
                    case TypeOrganelle.axon:
                        LastRemember = new Acson();
                        break;
                    case TypeOrganelle.mitoxondrion:
                        LastRemember = new Mitohondrion();
                        break;
                    case TypeOrganelle.vesicles:
                        LastRemember = new Vesicules();
                        break;
                    case TypeOrganelle.PSD:
                        LastRemember = new PSD();
                        break;
                    //case TypeOrganelle.membranes:                                                         /// Добавить после реализации
                    //    LastRemember = new Membranes();
                    //    break;

                    default:
                        throw new Exception("Неизвестный тип выбора органеллы в create_organel_Click");
                }


                //рисование в центре изображения
                LastRemember.NewPosition(Convert.ToInt32(g.VisibleClipBounds.Width) / 2,
                                         Convert.ToInt32(g.VisibleClipBounds.Height) / 2);
                LastRemember.Draw(g);

                pictureOrganelsBox.Image = imagePrew;
                pictureOrganelsBox.Refresh();

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        } 

        private void comboBoxViewType_SelectedIndexChanged(object sender, EventArgs e)     // Функция смены отображения
        {
            try
            {
                ComboBox comboBox = (ComboBox)sender;
                
                switch (comboBox.SelectedItem.ToString())
                {
                    case "image":
                        currImageViewType = TypeView.layer;
                        break;
                    case "mask axon":
                        currImageViewType = TypeView.maskAxon;
                        break;
                    case "mask mitoxondrion":
                        currImageViewType = TypeView.maskMitoxondrion;
                        break;
                    case "mask vesicules":
                        currImageViewType = TypeView.maskVesicules;
                        break;
                    case "mask PSD":
                        currImageViewType = TypeView.maskPSD;
                        break;
                    case "mask membranes":
                        currImageViewType = TypeView.maskMembranes;
                        break;
                    default:
                        throw new Exception("Неизвестный тип отображения в comboBoxViewType_SelectedIndexChanged");
                }

                Console.WriteLine(currImageViewType);
                Draw();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private void comboBoxOrganelsCreate_SelectedIndexChanged(object sender, EventArgs e)   // Функция смены типа создания объекта
        {
            try
            {
                ComboBox comboBox = (ComboBox)sender;

                switch (comboBox.SelectedItem.ToString())
                {
                    case "axon":
                        currCreateType = TypeOrganelle.axon;
                        break;
                    case "mitoxondrion":
                        currCreateType = TypeOrganelle.mitoxondrion;
                        break;
                    case "PSD":
                        currCreateType = TypeOrganelle.PSD;
                        break;
                    case "membrans":
                        currCreateType = TypeOrganelle.membranes;
                        break;
                    case "vesicules":
                        currCreateType = TypeOrganelle.vesicles;
                        break;
                    default:
                        throw new Exception("Неизвестный тип выбора органеллы для создания в comboBoxOrganelsCreate_SelectedIndexChanged");
                }

                Console.WriteLine(currCreateType);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private void listElements_SelectedIndexChanged(object sender, EventArgs e)                           /// исправить удаление на 2 клик
        {
            try
            {
                int deleteIndedex = listElements.SelectedIndex;
                if (deleteIndedex >= 0)
                {
                    Console.WriteLine($"delete {listElements.SelectedItem} with index {deleteIndedex}");
                    compartments.RemoveAt(listElements.SelectedIndex);
                    updateViewComponentList();
                    Draw();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}
