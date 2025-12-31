"""
Demo for calculation of the end probability for 'eval_pointing_param'
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import csv


NUM_TESTS = 6
EXPORT = False

demo_objects = [
    {
        "name":"011_banana_1",
        "color": "yellow",
        "confidence": 0.96,
        "position": [0.6293,-0.1834,0.5]
    },
    {
        "name":"011_banana_2",
        "color": "yellow",
        "confidence": 0.95,
        "position": [0.54309,-0.3126,0.5]
    },
    {
        "name":"003_tomato_sauce_1",
        "color": "red",
        "confidence": 0.94,
        "position": [0.927,-0.1661,0.5]
    },
    {
        "name":"003_tomato_sauce_2",
        "color": "red",
        "confidence": 0.93,
        "position": [0.608,-0.2733,0.5]
    },
    {
        "name":"010_apple_1",
        "color": "red",
        "confidence": 0.92,
        "position": [0.941,-0.2798,0.5]
    },
    {
        "name":"006_lemon_1",
        "color": "yellow",
        "confidence": 0.89,
        "position": [0.759,-0.2161,0.5]
    },
]


class RunReasonerTests():
  def __init__(self, action_param: str, num_tests: int,
               sigma=0.4, filename='', stats={}):
    assert action_param in ['left', 'right', 'shape', 'color'], "Incorrect action param!"
    
    self.action_param = action_param
    self.num_tests = num_tests
    self.sigma = sigma
    self.stats = stats
    assert demo_objects, "No 'demo_objects' need some DATA!"
    self.gdrn = demo_objects
    if len(stats.keys()) <= 0:
      for i in range(len(self.gdrn)):
        self.stats[i] = list()
    self.test_param()
    if EXPORT:
      self.export_dict_to_csv(f'{action_param}{filename}_stats.csv')
    
  def meet_ref_criteria(self, ref: dict, obj: dict, tolerance=0.0001):
    # return True if given object meets filter criteria of reference
    def check_shape(ref: dict, obj: dict) -> bool:
      ref_num = int(ref["name"][:3])
      obj_num = int(obj["name"][:3])
      return ref_num == obj_num
    
    def check_color(ref: dict, obj: dict) -> bool:
      ret = False
      if type(ref["color"]) == str and type(obj["color"]) == str:
        ret = (ref["color"] == obj["color"])
      elif type(ref["color"]) == list and type(obj["color"]) == str:
        ret = (obj["color"] in ref["color"])
      elif type(ref["color"]) == str and type(obj["color"]) == list:
        ret = (ref["color"] in obj["color"])
      elif type(ref["color"]) == list and type(obj["color"]) == list:
        for ref_color in ref["color"]:
          if ref_color in obj["color"]:
            ret = True
      return ret
    
    ret = False
    if (self.action_param == "color" and check_color(ref,obj)):
      ret = True
    elif (self.action_param == "shape" and check_shape(ref,obj)):
      ret = True
    elif self.action_param in ["left", "right"]:
      # print(f"Pointing vector: {dir_vector}")
      ref_pos = ref["position"]
      obj_pos = obj["position"]
      
      ref_to_pos = [obj_pos[0] - ref_pos[0], obj_pos[1] - ref_pos[1]]
      dot_product = (
        ref_to_pos[0] * -self.dir_vector[1] + 
        ref_to_pos[1] * self.dir_vector[0] )

      # print(f"{obj['name'] }, {obj['position'] },{dot_product}")
      if self.action_param == "right" and dot_product < -tolerance:
        ret = True
      elif self.action_param == "left" and dot_product > tolerance:
        ret = True
    
    # print(f"return: {ret}, ref, obj: {ref_num}, {obj_num}")
    return ret
  
  def evaluate_distances(self, distances: list, sigma=0.4):
      # returns list of probabilities from distances
      # distances to Gaussian distribution and normalized
      unnormalized_p = []
      for dist in distances:
        prob = np.exp(-(dist**2) / (2 * sigma**2))
        unnormalized_p.append(prob)
      normalized_p = unnormalized_p / np.sum(unnormalized_p)
      return list(normalized_p)
    
  def evaluate_reference(self, objects: list, ref_idx: int):
    # return list of probabilities for related objects of reference object
    ref = objects[ref_idx]
    dist_to_ref = []
    # firstly we calculate 1/distances to reference object
    for i in range(len(objects)):
      obj = objects[i]
      if self.meet_ref_criteria(ref,obj):
        dist = np.linalg.norm(
          np.array(obj["position"]) - np.array(ref["position"]) )
        if dist > 0:
          dist_to_ref.append(1/dist)
        else:
          dist_to_ref.append(dist)
      else:
        # for later sum unrelated objects needs to equal zero
        dist_to_ref.append(0.0)
        
    # compute prob from distances
    ref_probs = []
    if np.sum(dist_to_ref) != 0.0:
      ref_probs = np.array(dist_to_ref) / np.sum(dist_to_ref)
    else:
      ref_probs = list(np.zeros(len(objects)))
        
    return list(ref_probs)
  
  def evaluate_objects(self, objects: list, distances_from_line: list):
    dist_prob = self.evaluate_distances(distances_from_line,sigma=self.sigma)
    # print_array(dist_prob)
    # print("---")

    rows = []
    for idx in range(len(objects)):
      row = self.evaluate_reference(objects,idx)
      # print_array(row)
      # print("-")
      rows.append(list(dist_prob[idx]*objects[idx]["confidence"]*np.array(row)))
      # rows.append(row)
    
    prob_matrix = np.array(rows)
    # print_matrix(prob_matrix)
    ret = np.sum(prob_matrix,axis=0)
    # print_array(ret)
    # print()
    norm = np.sum(ret)
    return ret / norm
  
  def print_array(self, array):
    # print float array in readable way
    cnt = 0
    for number in array:
        cnt += 1
        if cnt <= len(array)-1:
            print("%.3f" % number,end=", ")
        else:
            print("%.3f" % number,end="")
    print()
    
  def print_matrix(self, matrix):
    for i in range(len(matrix)):
        print("[",end="")
        for j in range(len(matrix[0])):
            if j == len(matrix[0]) - 1:
                print(round(matrix[i][j],4), end="")
            else:
                print(round(matrix[i][j],4),end=", ")
        print("]",end="\n")
  
  def create_pointing(self, idx=-1, alfa=np.pi/3, beta=np.pi/18):
    
    def get_offsets(distance, alfa, beta):
      distance_p = np.sqrt(distance**2/(1+np.sin(alfa)**2))
      z = np.sin(alfa)*distance_p
      x = np.sqrt(distance_p**2/(1+np.sin(beta)**2))
      y = np.sin(beta)*x
      return np.array([x, y, z])
    
    if self.action_param in ['left', 'right']:
      beta = 0.0
    else:
      alfa = np.pi/2
    
    gdrn_names = [obj["name"] for obj in demo_objects]
    if idx == -1:
        idx = rnd.randint(0, len(gdrn_names)-1)
    else:
        idx %= 6
        # print(idx)
    position = np.array(self.gdrn[idx]["position"])
    wrist = position+get_offsets(0.2,alfa,beta)
    elbow = wrist+get_offsets(0.2,alfa,beta)
    
    return wrist, elbow

  def get_closest_to_line(self, max_dist=np.inf):
      
      def get_closest_point(line_points, test_point):
        p1, p2 = line_points
        p3 = test_point
        x1,y1,z1 = p1
        x2,y2,z2 = p2
        x3,y3,z3 = p3
        dx,dy,dz = x2-x1,y2-y1,z2-z1
        det = dx*dx+dy*dy+dz*dz
        a = (dy*(y3-y1)+dx*(x3-x1)+dz*(z3-z1))/det        
        return x1+a*dx, y1+a*dy, z1+a*dz
      
      distances_from_line = []
      for point in self.test_points:
        closest_point = np.array(get_closest_point(self.line_points,point))
        norm_dist = np.linalg.norm(closest_point-np.array(point))
        distances_from_line.append(norm_dist)
          
      if np.min(distances_from_line) > max_dist:
        return None, np.min(distances_from_line), distances_from_line
      return int(np.argmin(distances_from_line)), np.min(distances_from_line), distances_from_line
  
  def test_param(self):
    for i in range(self.num_tests):
      wrist, elbow = self.create_pointing(idx=i)
      self.dir_vector = list(wrist-elbow)
      v = 10*(wrist-elbow)
      self.line_points = [list(elbow),list(wrist+v)]
      self.test_points = [obj["position"] for obj in self.gdrn]
      pointing_id, _, distances_from_line = self.get_closest_to_line()

      out_probs = self.evaluate_objects(self.gdrn,distances_from_line)
      max_out = np.argmax(out_probs)
      self.stats[i%6].append(out_probs)
      
      # print(f"Input: action_param: {action_param}, pointing at: {demo_objects[pointing_id]["name"]}")
      # print(self.gdrn[max_out]["name"],end="\t")
      # self.print_array(out_probs)
              
  def export_dict_to_csv(self, filename):
    """
    Exports a dictionary with keys mapping to lists of vectors to a CSV file.

    :param data_dict: Dictionary where keys are integers and values are lists of vectors (each vector is a list of numbers).
    :param file_name: Name of the CSV file to create (e.g., "output.csv").
    """
    with open(filename, mode='w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      
      for key, vectors in self.stats.items():
        # Write a header for the key
        writer.writerow([f"Key {key}"])
        
        # Write each vector in the list
        for vector in vectors:
            writer.writerow(vector)
        
        # Add an empty row for separation
        writer.writerow([])
    
    print(f"Data successfully exported to {filename}")
    
    
def plot_dict_graph(data_dict,param,sigma_values):
    """
    Creates a graph from a dictionary structure with keys mapping to lists of vectors.

    :param data_dict: Dictionary where keys are integers and values are lists of vectors (each vector is a list of numbers).
    """
    for key in data_dict:
      key_value = np.array(data_dict[key]).T
      data_dict[key] = key_value
    
    
    # Check if the number of sigma values matches vector length
    if not all(len(vectors[0]) == len(sigma_values) for vectors in data_dict.values()):
        raise ValueError("The length of vectors must match the number of sigma values (10).")

    # Prepare the plot
    labels = ['banana_1','banana_2','tomato_soup_1','tomato_soup_2','apple_1', 'lemon_1']
    markers = ['o','^','s','v','d','x']
    # Create a graph for each key
    for key, vectors in data_dict.items():
      plt.figure(figsize=(6.7,6))

      for i, vector in enumerate(vectors):
        # Plot the vector against the sigma values
        plt.plot(sigma_values, vector,
                  label=labels[i],
                  marker=markers[i],
                  linestyle='-')

      # Add labels, title, and legend
      plt.xlim([np.min(sigma_values),np.max(sigma_values)])
      # plt.ylim([-0.05,0.6])
      plt.xlabel(r"Standardní odchylka ukazování $\sigma_d$", fontsize=12)
      plt.ylabel(r"Výsledná pst. $\mathrm{T}_i$", fontsize=12)
      plt.title(f"Akční parametr: '{param}', ukazuje se na {labels[key]}", fontsize=12)
      plt.legend(loc='lower left', fontsize=8)

      # Add grid and improve layout
      plt.gca().invert_xaxis()
      plt.grid(True, which='both', linestyle='--', linewidth=0.5)
      plt.tight_layout()

      # Show the plot for this key
      plt.show()


if __name__ == "__main__":
  shape_d = {}
  color_d = {}
  right_d = {}
  left_d = {}
  sigmas = np.arange(0.4, 0.0, -0.05)
  for sigma in sigmas:
    assert sigma > 0, "It leads to death, to divide by zero"
    # print(f'sigma: {sigma}')
    RunReasonerTests('shape',NUM_TESTS,sigma,f'_s{sigma}',shape_d)
    RunReasonerTests('color',NUM_TESTS,sigma,f'_s{sigma}',color_d)
    RunReasonerTests('right',NUM_TESTS,sigma,f'_s{sigma}',right_d)
    RunReasonerTests('left',NUM_TESTS,sigma,f'_s{sigma}',left_d)
  
  
  # plot_dict_graph(shape_d,'shape',sigmas)
  # plot_dict_graph(color_d,'color',sigmas)
  # plot_dict_graph(right_d,'right',sigmas)
  # plot_dict_graph(left_d,'left',sigmas)