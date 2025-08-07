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
    def __init__(self,filepath,filetype,k = 32,tolerance = 0.0001,kn = 4):
        self.filepath = filepath
        self.tolerance = tolerance
        self.RGBUnit_list = []
        self.RGB_color_clump = dict()
        self.centroid_list = []
        self.trained_cluster_dictionary = dict()
        self.k=k
        self.kn = kn

        if filetype == "csv":self.digest_csv(filepath)
        elif filetype == "unclass": self.digest_unclassified_RGB(filepath)
        elif filetype == "class": self.digest_classified_RGB(filepath)
    

    #Takes in an unclassified .bin file and then generates clusters out of it. From that it creates an actual classified .bin file
    #tested!
    def digest_unclassified_RGB(self,filepath):
        with open(filepath, 'rb') as binfile:
            while True:
                bytes_read = binfile.read(4)
                if not bytes_read or len(bytes_read) < 4:
                    break
                r, g, b, color_num = bytes_read

                #create a RGB object here:
                self.RGBUnit_list.append(RGBUnit(r,g,b,self.get_color_label(color_num)))
        self.K_Means()
        self.generate_classified_RGB(self.RGB_color_clump)
            
    def printRGBUnit_list(self):
        for i in self.RGBUnit_list:
            print(i.r,i.g,i.b,i.label.value[0])
    
    #tested!
    def number_miniheader(self,number):
        primer_byte = bytes([int(number/255) + 1])

        byte_result = primer_byte + number.to_bytes(int(number/255) + 1,byteorder='big')

        return byte_result
    
    def miniheader_decrypt(self,number_byte):
        number_description = list(number_byte)

        primer_number = number_description[0]
        self.aggregate_byte_list(number_description[1:])

    #tested!(again)
    #generates the master header. At best should just be two bytes.
    # such a list of bytes looks like: primer byte to describe number of the next bytes, bytes to describe the number of clusters.
    def generate_master_header(self,cluster_dictionary):
        
        number_of_clusters = len(cluster_dictionary.keys())
        number_of_bytes = int(number_of_clusters / 255) + 1

        byte_header_1 = bytes([number_of_bytes])
        


        master_header = byte_header_1 +  number_of_clusters.to_bytes(number_of_bytes,byteorder='big')

        #generate the index table

        raw_cluster_data = []
        cluster_index = []
        offset_from_start_of_raw_cluster = 0
        offset_from_current_cluster_index = 0
        

        for k,v in cluster_dictionary.items():
            cluster_header = self.generate_cluster_header(k,v)
            raw_cluster_data.append(cluster_header)
            offset_from_start_of_raw_cluster += len(cluster_header)
        
        
        for cluster_byte in reversed(raw_cluster_data):
            offset_from_start_of_raw_cluster -= len(cluster_byte)
            cluster_index_chunk = cluster_byte[:3] + self.number_miniheader(offset_from_start_of_raw_cluster + offset_from_current_cluster_index)
            offset_from_current_cluster_index += len(cluster_index_chunk)
            cluster_index.insert(0,cluster_index_chunk)

        offset_from_true_start = len(master_header)
        

        for cluster_index_number in range(len(cluster_index)):
            offset_from_true_start += len(cluster_index[cluster_index_number])
            
            centroid_point = cluster_index[cluster_index_number][:3]

            
            
            index_number = self.aggregate_byte_list(list(cluster_index[cluster_index_number][4:]))

            new_index_number = index_number + offset_from_true_start
            #print(new_index_number)

            #we replace the current centroid index byte with the updated one to accuractly reflect as an offset from the start of the .bin file.
            cluster_index[cluster_index_number] = centroid_point + self.number_miniheader(new_index_number)
        
        byte_result = master_header
        cluster_index += raw_cluster_data

        for i in cluster_index:
            byte_result+=i
        
        

        return byte_result
    

    #Returns an ordered collection of bytes that represent a singular cluster.
    #such a list of bytes looks like: primer byte, bytes to describe number of data points, r, g, b, 4 bytes per data point
    #tested! (again)
    def generate_cluster_header(self,key,value):
        number_of_points = len(value)
        number_of_bytes = int(number_of_points/255) + 1

        byte_header_1 = bytes([number_of_bytes])

        byte_result = byte_header_1 + number_of_points.to_bytes(number_of_bytes,byteorder='big')

        centroid_bytes = bytes(list(key))

        byte_result = centroid_bytes + byte_result
        

        for i in value:
            point_byte_list = list(i)
            
            point_byte_list[3] = self.get_color_number(point_byte_list[3])
            

            point_byte_list = bytes(point_byte_list)
            byte_result += point_byte_list
        return byte_result
    
    #Generates the classified .bin file that stores the cluster information.
    #not tested!
    def generate_classified_RGB(self,cluster_dictionary):
        with open('classified.bin','wb') as binfile:
            binfile.write(self.generate_master_header(cluster_dictionary))
    
    #takes a list of bytes represented as an integer list and returns the aggregated number being represented by it.
    #tested
    def aggregate_byte_list(self,byte_list):
        result = ""
        for i in byte_list:
            result += bin(i)[2:]
        return int(result,2)

    #reads a classified rgb .bin file and compiles it into a dictionary field for ease of use
    #dictionary field is self.trained_cluster_dictionary
    #We will then connect this with K-nearest neighbors
    #not tested
    def digest_classified_RGB(self,filepath):
        with open(filepath,'rb') as binfile:
            
            primer_number = list(binfile.read(1))[0]

            total_number_of_clusters = self.aggregate_byte_list(list(binfile.read(primer_number)))

            for i in range(total_number_of_clusters):#we iterate as many times we we have clusters
                cluster_primer_number = list(binfile.read(1))[0]
                total_number_of_data_points = self.aggregate_byte_list(list(binfile.read(cluster_primer_number)))
                centroid = tuple(list(binfile.read(3)))
                self.trained_cluster_dictionary[centroid] = set()
                for j in range(total_number_of_data_points):
                    rgb_point = list(binfile.read(4))
                    rgb_point[3] = self.get_color_label(rgb_point[3]).value[0]
                    self.trained_cluster_dictionary[centroid].add(tuple(rgb_point))


            
                

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

    #takes in a .csv file with rgb data points and generates an unclassified .bin file of those rgb data points
    #tested and should work
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
        
        total_inertia = 0


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
            #this inner for loop combs through centroid column vector to see if we can update the min distance of any point to any centroid.
            for j in range(len(temp_column_vector)):
                if temp_column_vector[j] < centroid_column_vector[j,0]:
                    centroid_column_vector[j,0] = temp_column_vector[j][0]
                    centroid_column_vector[j,1] = i + 1
            output_centroid_list.append(next_centroid)
        for data_point in centroid_column_vector:
            total_inertia += data_point[0]
        
            

        
        
        #re-add the color values
        cluster_dictionary = dict()
        
        for i in range(len(centroid_column_vector)):
            cluster_dictionary.setdefault(output_centroid_list[int(centroid_column_vector[i][1])], set()).add(rgb_list[i])


        return cluster_dictionary,total_inertia

    #every entry of column_vector is: [distance from closest centroid, index of closest centroid]
    #tested!
    def prob_distribute(self,column_vector):
        total_weight = np.sum(column_vector[:, 0])
        if total_weight == 0:print(column_vector)

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
    

    #tested!
    #returns a new cluster dictionary and total inertia of that dictionary
    def calculate_new_centroids(self,current_dict,rgb_list):
        total_inertia = 0
        
        current_centroid_list = []
        

        #This for loop calculates the new centroid for every cluster.
        for cluster in current_dict.values():
            arr = np.array([t[:-1] for t in cluster])
            new_centroid = tuple(int(round(x)) for x in np.mean(arr,axis=0).tolist()) 
            current_centroid_list.append(new_centroid)
        
        
        #Turns the centroid list and rgb list into np arrays. 
        centroid_array = np.array(current_centroid_list)
        #data_point_array is rgb_list ordered
        data_point_array = np.array([row[:-1] for row in rgb_list])


        #Calculates every point's distance from every cluster
        distance_matrix = cdist(data_point_array,centroid_array,metric='sqeuclidean') #rgb_list ordered!
        #we collapse the matrix to get their smallest distance from a centroid.
        distance_column_vector = np.argmin(distance_matrix, axis=1) #rgb_list ordered!
        #np.argmin relies on the previously ordered distance_matrix with numbers as index values pointing back to distance_matrix




        new_cluster_dictionary = dict()

        #distance_column_vector[rgb_point_index] = centroid_list index
        for rgb_point_index in range(len(distance_column_vector)):
            total_inertia += distance_matrix[rgb_point_index][distance_column_vector[rgb_point_index]]
            new_cluster_dictionary.setdefault(current_centroid_list[distance_column_vector[rgb_point_index]], set()).add(rgb_list[rgb_point_index])

        return new_cluster_dictionary,total_inertia
    
        


    #Applies the K_Means algorithm. Result is in self.RGB_color_clump
    def K_Means(self):
        #data_point_list is ordered list of triple lists (r,g,b,"label") which are number values
        data_point_list = []
        for i in self.RGBUnit_list:
            temp_list = (i.r,i.g,i.b,i.label.value[0])
            data_point_list.append(temp_list)
        

        self.RGB_color_clump,current_inertia = self.init_centroid_points(data_point_list,self.k)

        new_inertia = 0 #diff between current and new will always be more than self.tolerance

        while True:
            print(current_inertia,"-",new_inertia,"=",abs(current_inertia - new_inertia))
            #new_inertia is generated a new number
            self.RGB_color_clump,new_inertia = self.calculate_new_centroids(self.RGB_color_clump,data_point_list)

            if abs(current_inertia - new_inertia) <= self.tolerance:break

            current_inertia = new_inertia
    
    #data_point is a tuple of (r,g,b)
    def K_Nearest(self,data_point):
        if len(list(self.trained_cluster_dictionary)) == 0:
            self.digest_classified_RGB(self.filepath)

        

unit = RGB_Graph("output.bin","unclass",2)

print(unit.RGB_color_clump)

test_cluster_dictionary = {
    (1,2,3):{(5,5,5,"Black"),(6,7,3,"Black")},
    (4,3,2):{(1,2,1,"White"),(9,6,3,"White"),(93,4,34,"Grey")},
    (65,23,12):{(1,2,2,"Yellow"),(65,45,23,"Grey"),(34,123,12,"Red"),(90,78,67,"Blue"),(123,12,12,"Red")}
}



#file_byte_stream = master_header + cluster_index_data + raw_cluster_data

#print(file_byte_stream)



# test_list = [
#     (5,5,5,"Black"),
#     (6,7,3,"Black"),
#     (11,7,8,"Black"),
#     (10,5,4,"Black"),
#     (100,100,100,"Grey"),
#     (110,90,120,"Grey")
# ]


