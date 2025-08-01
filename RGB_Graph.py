from RGB import RGBUnit
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import random

import numpy as np
from scipy.spatial.distance import cdist


class RGB_Graph:
    
    

    #we will use this to read 
    def __init__(self,filepath,csv = False):
        self.RGBUnit_list = []
        self.RGB_color_clump = dict()
        self.centroid_list = []

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
    

    #tested!
    #Takes a raw list of rgb data points as such: [r,g,b,color_value] and variable k to generate a dictionary of clusters(and centroids)
    def init_centroid_points(self,rgb_list,k):

        fixed_data_point_array = np.array([row[:-1] for row in rgb_list]) #we exclude the color_value for ease of calculation with cdist
        


        ran_index = random.randint(0,len(rgb_list)-1)
        current_centroid = tuple(fixed_data_point_array[ran_index].tolist()) #our very random first centroid point initialized.


        output_centroid_list = [] #This is what we're going to output
        output_centroid_list.append(current_centroid)


        #We store the min distance to any centroid in this column vector here
        centroid_column_vector = cdist(fixed_data_point_array, np.array([current_centroid]), metric='sqeuclidean').tolist()
        for i in centroid_column_vector:
            i.append(0)
        centroid_column_vector = np.array(centroid_column_vector) #we updating min distance from any centroid here.

        
        

        for i in range(k-1):
            
            next_centroid_index = self.prob_distribute(centroid_column_vector) #use prob distribution to select our next centroid

            next_centroid = tuple(fixed_data_point_array[next_centroid_index].tolist()) #should be an array of 3 [r,g,b]


            #calculate distance from all points to new centroid
            temp_column_vector = cdist(fixed_data_point_array,np.array([next_centroid]),metric='sqeuclidean') 

            #both temp and centroid column vector are ordered so they always match up.
            #this function combs through centroid column vector to see if we can update the min distance of any point to any centroid.
            for j in range(len(temp_column_vector)):
                if temp_column_vector[j] < centroid_column_vector[j,0]:
                    centroid_column_vector[j,0] = temp_column_vector[j][0]
                    centroid_column_vector[j,1] = i + 1
            output_centroid_list.append(next_centroid)
            

        
        
        #re-add the color values
        cluster_dictionary = dict()
        
        for i in range(len(centroid_column_vector)):
            cluster_dictionary.setdefault(output_centroid_list[int(centroid_column_vector[i][1])], set()).add(rgb_list[i])


        return cluster_dictionary

    #every entry of column_vector is: [distance from closest centroid, index of closest centroid]
    #tested!
    def prob_distribute(self,column_vector):
        total_weight = np.sum(column_vector[:, 0])

        distance_column_vector = column_vector[:,0]
        
        select_random = random.random()

        current_marker = 0.0

        selected_data_point_index = 0

        for iterator in range(len(distance_column_vector)):
            current_marker += distance_column_vector[iterator]/total_weight
            if current_marker > select_random:
                selected_data_point_index = iterator
                break
        return selected_data_point_index
    
    def calculate_new_centroids(self,current_dict):
        current_centroid_list = []

        for cluster in current_dict.values():
            pass


    
    def K_Means(self):
        #data_point_list is ordered list of triple lists (r,g,b) which are number values
        data_point_list = []
        for i in self.RGBUnit_list:
            temp_list = (i.r,i.g,i.b,i.label.value[0])
            data_point_list.append(temp_list)
        

        self.RGB_color_clump = self.init_centroid_points(data_point_list,32)
        
        
        #add a loop

        self.RGB_color_clump = self.calculate_new_centroids(self.RGB_color_clump)




unit = RGB_Graph("output.bin")

test_list = [
    (5,5,5,"color1"),
    (6,7,3,"color1"),
    (11,7,8,"color2"),
    (10,5,4,"color2"),
    (100,100,100,"color3"),
    (110,90,120,"color3")
]
output = unit.init_centroid_points(test_list,3)

print(output)



# test_column_vector = [
#     [75,0],
#     [0,0],
#     [0,0],
#     [25,0]
# ]


# prob_dictionary = dict()
#unit.prob_distribute(np.array(test_column_vector))

# for i in range(1000):
#     distributed_selected_value = unit.prob_distribute(np.array(test_column_vector))
#     prob_dictionary[distributed_selected_value] = prob_dictionary.get(distributed_selected_value,0) + 1
# print(prob_dictionary)


#unit.K_Means()


