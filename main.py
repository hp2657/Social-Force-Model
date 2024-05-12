import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

  
'''parameter'''
'''
x_min, x_max, y_min, y_max, initial_speed_x, initial_speed_y
initial_speed_x, initial_speed_y should be a vector with length = 1
'''
departure_region = {
    'a': [-8,-8,-0.5,0.5,1.1,0]
    }

destination_region = {
    'a': [13,13,0,0,0,1]
    }

storage_position = np.array([-10, -10])

obs = [[0, 10, 1, 1], 
       [0, 10, -1, -1],
       [10, 10, 0.5, 1],
       [10, 10, -1, -0.5],
       [-15, 0, 5, 5],
       [-15, 0, -5, -5],
       [0, 0, 1, 5],
       [0, 0, -5, -1],
       [-15, -15, -5, 5]
       ]

step = 6000

number_of_people = 800

initial_vel = 1.1

max_vel = 2

mean_weight = 75

std_weight = 5

release_system_parameter = {
    'num_of_people_keep': 100,
    'release_time_interval': 10
    }
    
def pedestrain_generator(departure_region, destination_region, total_time, num_of_pedestrain, 
                         initial_vel, max_vel, mean_weight, std_weight):
    initial_state_data = list()
    
    for i in range(num_of_pedestrain):
        departure_region_temp_code = random.sample(list(departure_region.keys()), 1)[0]
        departure_region_temp = departure_region[departure_region_temp_code]
        departure_region_x_lowerbound = departure_region_temp[0]
        departure_region_x_upperbound = departure_region_temp[1]
        departure_region_y_lowerbound = departure_region_temp[2]
        departure_region_y_upperbound = departure_region_temp[3]
        departure_position_x = random.uniform(departure_region_x_lowerbound, departure_region_x_upperbound)
        departure_position_y = random.uniform(departure_region_y_lowerbound, departure_region_y_upperbound)
        
        destination_region_temp = random.sample(list(destination_region.values()), 1)[0]
        while destination_region_temp == departure_region_temp:
            destination_region_temp = random.sample(list(destination_region.values()), 1)[0]
        destination_region_x_lowerbound = destination_region_temp[0]
        destination_region_x_upperbound = destination_region_temp[1]
        destination_region_y_lowerbound = destination_region_temp[2]
        destination_region_y_upperbound = destination_region_temp[3]
        destination_position_x = random.uniform(destination_region_x_lowerbound, destination_region_x_upperbound)
        destination_position_y = random.uniform(destination_region_y_lowerbound, destination_region_y_upperbound)
        
        velocity = random.gauss(initial_vel, 0.3)
        velocity_x = velocity * departure_region_temp[4]
        velocity_y = velocity * departure_region_temp[5]
        
        max_velocity = random.gauss(max_vel, 0.3)
        
        weight = random.gauss(mean_weight, std_weight)
        
        pedestrain_data = [departure_position_x, 
                           departure_position_y,
                           velocity_x,
                           velocity_y,
                           destination_position_x,
                           destination_position_y,
                           max_velocity,
                           weight,
                           0]
        initial_state_data.append(pedestrain_data)
        
    return initial_state_data


class Simulator:
    
    def __init__(self, state, storage_position, release_system_parameter, obstacles=None):
        self.state = state
        self.storage_position = storage_position
        self.avg_velocity = []
        self.avg_distance = []
        self.avg_force = []
        self.social_force = 0
        self.state_record = []
        self.obstacles = obstacles
        self.release_system_parameter = release_system_parameter
        self.count_down = 0
        self.processing_state = []
        self.processing_sign = []
        
    def release_system(self):
        num_of_running = sum(self.state[:, 8] == 1)
        if self.count_down != 0:
            self.count_down = self.count_down - 1
        else:
            if num_of_running < release_system_parameter['num_of_people_keep']:
                next_to_release = np.where(self.state[:, 8] == 0)[0]
                if len(next_to_release) > 0:
                    self.state[next_to_release[0], 8] = 1
                    self.count_down = release_system_parameter['release_time_interval']
        self.processing_state = self.state[self.state[:, 8] == 1]
        self.processing_sign = self.state[:, 8] == 1
                
     
    def compute_forces(self):
        self.force = Force(self.processing_state, self.obstacles)
        return self.force.accelaration()


    def step_once(self):
        self.release_system()
        if self.processing_state.shape[0] == 0:
            return
        self.peds = PedState(self.processing_state, self.processing_state[:,6], storage_position)
        self.processing_state = self.peds.step(self.compute_forces())
        self.state[self.processing_sign] = self.processing_state
        temp_state = self.state.copy()
        self.state_record.append(temp_state)
        

    def step(self, n=1):
        for _ in range(n):
            self.step_once()
            if _ % 50 == 0:
                print(_/n * 100)
        return self

class PedState:

    def __init__(self, state, max_speeds, storage_position):
        self.step_width = 0.01
        self.max_speeds = max_speeds
        self.release_system_parameter = release_system_parameter
        self.storage_position = storage_position
        self.update(state)

    def update(self, state):
        self.state = state

    def pos(self):
        return self.state[:, 0:2]

    def vel(self):
        return self.state[:, 2:4]
    
    def goal(self):
        return self.state[:, 4:6]

    def step(self, force, groups=None):

        desired_velocity = self.vel() + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        desired_velocity[self.close_to_goal()] = np.array([0, 0])

        next_state = self.state
        next_state[:, 0:2] += desired_velocity * self.step_width
        next_state[:, 2:4] = desired_velocity
        next_state[:, 0:2][self.close_to_goal()] = self.storage_position
        next_state[:, 0:2][self.check_inside(next_state[:, 0:2])] = self.storage_position
        next_state[:, 8][self.check_lines_equal(next_state[:, 0:2], self.storage_position)] = 2   
        
        self.update(next_state)

        return self.state

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)
    
    def close_to_goal(self):
        position = self.pos()
        goal = self.goal()
        distance = np.linalg.norm(position - goal, axis = 1)
        return distance < 1
    
    def check_inside(self, next_state):
        check_inside_x = abs(next_state[:, 0] - self.storage_position[0]) < 1
        check_inside_y = abs(next_state[:, 1] - self.storage_position[1]) < 1
        check_inside = check_inside_x & check_inside_y
        return check_inside
    
    def check_lines_equal(self, arr, target):
        return np.all(arr == target, axis=1)


class Force:
    def __init__(self, state, obstacle):
        self.state = state
        self.obstacle = obstacle
        self.max_vel = state[:, 6]
        self.r_ij = 0.35
        self.A_i = 2000
        self.B_i = 0.08
        self.k = 120000
        self.gaba = 240000
        self.size = self.state.shape[0]
        self.weight = self.state[:, 7]
        
    def position(self):
        return self.state[:, 0:2]
    
    def velocity(self):
        return self.state[:, 2:4]
    
    def goal(self):
        return self.state[:, 4:6]
    
    @staticmethod
    def normalize_vectors(vectors):
        
        norm_factors = []
        for line in vectors:
            norm_factors.append(np.linalg.norm(line))
        norm_factors = np.array(norm_factors)
        normalized = vectors / np.expand_dims(norm_factors, -1)
        for i in range(norm_factors.shape[0]):
            if norm_factors[i] == 0:
                normalized[i] = np.zeros(vectors.shape[1])
        return normalized, norm_factors
    
    @staticmethod
    def normalize_single_vectors(vector):
        norm_factor = np.linalg.norm(vector)
        if norm_factor == 0:
            return np.array([0, 0])
        else:
            return vector / norm_factor
    
    @staticmethod
    def delta_v_ji(x, y):
        temp = []
        for i in range(x.shape[0]):
            temp.append(np.matmul(x[i], y[i].reshape(2,-1))[0])
        return np.array(temp)
    
    @staticmethod
    def vec_diff(vector):
        diff = np.expand_dims(vector, 1) - np.expand_dims(vector, 0)
        return diff
    
    
    def each_diff(self, vector):
        diff = self.vec_diff(vector)
        diff = diff[
            ~np.eye(diff.shape[0], dtype=bool), :
        ]

        return diff
    
    @staticmethod
    def g(x):
        x[ x<0 ] = 0
        return x
    
    @staticmethod
    def g_single(x):
        if x < 0:
            return 0
        else:
            return x
    
    def desired_force(self):
        relexation_time = 0.5
        pos = self.position()
        vel = self.velocity()
        goal = self.goal()
        direction, dist = self.normalize_vectors(goal - pos)
        force = (direction * self.max_vel[:, np.newaxis] - vel)/relexation_time
        return force
    
    def social_force(self):
        if self.state.shape[0] == 1:
            return np.array([0, 0])
        pos_diff = self.each_diff(self.position())
        vel_diff = self.each_diff(self.velocity())
        n_ij, d_ij = self.normalize_vectors(pos_diff)
        t_ij = np.copy(n_ij)
        t_ij[:,[0,1]]=t_ij[:,[1,0]]
        t_ij[:,0] = -t_ij[:,0]
        vel_diff = -vel_diff
        delta_v_ji_value = self.delta_v_ji(vel_diff, t_ij)
        part_1 = self.A_i * np.exp( ( self.r_ij - d_ij )/self.B_i ) + self.k * self.g(self.r_ij - d_ij)
        part_1 = part_1.reshape(part_1.shape[0], -1)
        part_2 = self.gaba * self.g(self.r_ij - d_ij) * delta_v_ji_value
        part_2 = part_2.reshape(part_2.shape[0], -1)
        force = part_1 * n_ij + part_2 * t_ij
        
        force = np.sum(force.reshape((self.size, -1, 2)), axis=1)
        
        return force
    
    def wall_distance(self, point, line_start, line_end):
        line_vector = line_end - line_start
        point_line_vector = point - line_start
        intercept_point = line_start + line_vector * (np.dot(line_vector, point_line_vector) / np.dot(line_vector, line_vector))
        x_min = min(line_start[0], line_end[0])
        x_max = max(line_start[0], line_end[0])
        y_min = min(line_start[1], line_end[1])
        y_max = max(line_start[1], line_end[1])
        point_on_line = intercept_point[0] <= x_max and intercept_point[0] >= x_min and intercept_point[1] <= y_max and intercept_point[1] >= y_min
        distance = np.linalg.norm(point-intercept_point)
        direction = point-intercept_point
        if not(point_on_line):
            return False, False
        elif distance > 0.175:
            return False, False
        else:
            return distance, direction
        
    def compute_force(self, position, obstacles, velocity):
        force = np.array([0,0])
        for obstacle in obstacles:
            obstacle_start = np.array([obstacle[0], obstacle[2]])
            obstacle_end = np.array([obstacle[1], obstacle[3]])
            distance, direction = self.wall_distance(position, obstacle_start, obstacle_end)
            if distance:
                t_iw = obstacle_end - obstacle_start
                t_iw = self.normalize_single_vectors(t_iw)
                n_iw = np.copy(t_iw)
                n_iw = np.array([-n_iw[1], n_iw[0]])
                if np.dot(n_iw, direction) < 0:
                    n_iw = -n_iw
                part_1 = (self.A_i * np.exp((0.35 - distance)/self.B_i)+self.k * self.g_single(0.35 - distance)) * n_iw
                part_2 = self.k * self.g_single(0.35 - distance) * np.dot(velocity, t_iw) * t_iw
                force = force + part_1 + part_2
                
        return force
                    
    
    def wall_interaction_force(self):
        pos = self.position()
        vel = self.velocity()
        force = [self.compute_force(position, self.obstacle, velocity) for position, velocity in zip(pos, vel)]
        force = np.array(force)
        
        return force
    
    def accelaration(self):
        force = self.social_force() + self.wall_interaction_force()
        accelaration = force / self.weight[:, np.newaxis] + self.desired_force()
        return accelaration





    
class SceneAnimation:
    def __init__(self, state_record, obstacle, file_name):
        self.state_record = state_record
        self.obstacle = obstacle
        self.file_name = file_name + '.mp4'
        self.frames = len(state_record)
        self.fig, self.ax = plt.subplots()
        self.points, = self.ax.plot([], [], 'bo')
        self.frame_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10, color='black')
        #self.ax.set_xlim((-20, 15))
        #self.ax.set_ylim((-10, 10))
        
    def update(self, frame):
        current_state = self.state_record[frame]
        x_data = current_state[:, 0]
        y_data = current_state[:, 1]
        
        self.points.set_data(x_data, y_data)
        
        self.frame_text.set_text('Frame: {}'.format(frame))
        
        return self.points, self.frame_text
    
    def animate(self):
        print('making animation......')
        for obstacles in self.obstacle:
            self.ax.plot([obstacles[0], obstacles[1]], [obstacles[2], obstacles[3]], color = 'black')
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.frames, blit=True)
        #ani.save(self.file_name, fps = 50, writer='ffmpeg')
        ani.save(self.file_name, fps=100, writer='ffmpeg', codec='h264', bitrate=-1)
        return ani   
    
    

class TrajectoryShow:
    def __init__(self, state_record, obstacle, file_name, k):
        self.state_record = state_record
        self.obstacle = obstacle
        self.file_name = file_name + '.jpg'
        self.fig, self.ax = plt.subplots()
        self.k = k

    def extract(self, k):
        kth_rows = np.array([array[k] for array in self.state_record])
        return kth_rows

    def plot(self):
        for obstacles in self.obstacle:
            self.ax.plot([obstacles[0], obstacles[1]], [obstacles[2], obstacles[3]], color='black')
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.k)))
        color_list = [mcolors.to_hex(color) for color in colors]
        for index, k in enumerate(self.k):
            trajectory_data = self.extract(k)
            x_data, y_data = trajectory_data[:, 0], trajectory_data[:, 1]
            self.ax.scatter(x_data, y_data, color=color_list[index], s=0.01, label=f'k={k}')
        self.ax.legend()
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.savefig(self.file_name, dpi=1200)




    
    
    

if __name__ == "__main__":
    data = pedestrain_generator(departure_region, destination_region, step, number_of_people, initial_vel, 
                                max_vel, mean_weight, std_weight)
    data = np.array(data) 
    initial_state = data
    obs = obs
    s = Simulator(
        initial_state,
        storage_position,
        release_system_parameter,
        obstacles=obs
    )
    
    s.step(step)
    sa =  SceneAnimation(s.state_record, obs, 'animation')
    sa.animate()
    point_index = [121, 122, 123, 124, 125]
    tc = TrajectoryShow(s.state_record, obs, 'trajectory', point_index)
    tc.plot()
    

