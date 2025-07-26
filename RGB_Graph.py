from RGB import RGBUnit
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



class RGB_Graph:
    
    

    #we will use this to read 
    def __init__(self,filepath,csv = False):
        self.RGBUnit_list = []
        self.RGB_color_clump = dict()

        if csv == False:
            self.digest_unclassified_RGB(filepath)
        else:
            self.digest_csv(filepath)
        

    def digest_unclassified_RGB(self,filepath):
        with open(filepath, 'rb') as binfile:
            while True:
                bytes_read = binfile.read(4)
                if not bytes_read or len(bytes_read) < 4:
                    break
                r, g, b, color_num = bytes_read
                #print(f"R: {r}, G: {g}, B: {b}, Color Number: {color_num}")
                #create a RGB object here:
                self.RGBUnit_list.append(RGBUnit(r,g,b,self.get_color_label(color_num)))
                

    #returns the number based on the color name
    def get_color_number(self,color_name):
        for color in RGBUnit.ColorFormat:
            if color.value[0] == color_name:
                return color.value[1]
        return None
    
    #returns the enum object based on number
    def get_color_label(self,color_number):
        for color in RGBUnit.ColorFormat:
            if color.value[1] == color_number:
                return color
        return None

    def digest_csv(self,filepath):
        data = []
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)

            header = next(csv_reader)
            for row in csv_reader:
                data.append(row)
        
        with open('output.bin', 'wb') as binfile:
            #write the header here NOT FUNISHED YET

            for row in data:
                bin_row = [int(x) for x in row[:3]]
                bin_row.append(self.get_color_number(row[3]))
                bin_row = bytes(bin_row)
                binfile.write(bin_row)


    def plot_3D(self):
        LABEL_COLORS = {
            "Red":     (1.0, 0.0, 0.0),   
            "Blue":    (0.0, 0.0, 1.0),   
            "Green":   (0.0, 1.0, 0.0),   
            "Yellow":  (1.0, 1.0, 0.0),  
            "Orange": (1.0, 0.647, 0.0), 
            "Purple":  (0.5, 0.0, 0.5),    
            "Pink": (1.0, 0.753, 0.796), 
            "Brown": (0.545, 0.271, 0.075),
            "Grey": (0.502, 0.502, 0.502),
            "Black": (0.0, 0.0, 0.0),
            "White": (1.0,1.0,1.0)
            }
        
        for i in self.RGBUnit_list:
            
            self.RGB_color_clump[i.label.value[0]] = self.RGB_color_clump.get(i.label.value[0],[])
            r = i.r
            g = i.g
            b = i.b
            self.RGB_color_clump[i.label.value[0]].append([r,g,b])
            
        
        color_labels = dict()
        for k,v in self.RGB_color_clump.items():
            color_labels[k] = [k] * len(v)
        all_rgb_data = []


        all_labels = []
        for color_name,color_clump in self.RGB_color_clump.items():
            all_labels += color_labels[color_name]
            clump_array = np.array(color_clump)
            all_rgb_data.append(clump_array)
        all_rgb_data = np.vstack(all_rgb_data)

        

        r_values = all_rgb_data[:, 0]
        g_values = all_rgb_data[:, 1]
        b_values = all_rgb_data[:, 2]

        plot_colors = [LABEL_COLORS[label] for label in all_labels]
        

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(r_values, g_values, b_values, c=plot_colors, marker='o', s=50)

        ax.set_xlabel('Red (R)')
        ax.set_ylabel('Green (G)')
        ax.set_zlabel('Blue (B)')
        ax.set_title('3D Representation of RGB Data Points (Color-Coded by Label)')

        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])

        ax.grid(True)
        ax.view_init(elev=20, azim=120)

        plt.show()
        




unit = RGB_Graph("output.bin")

unit.plot_3D()
#unit.digest_csv("colors.csv")

