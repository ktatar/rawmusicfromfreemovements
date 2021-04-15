import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation 
import mpl_toolkits.mplot3d as plt3d
        
class PoseAnimPlayer:
    def __init__(self, joint_data, edge_data, axis_min, axis_max):
        self.joint_data = joint_data
        self.edge_data = edge_data
        self.start_frame = 0
        self.end_frame = min(joint_data.shape[0], 100)
        self.axis_min = axis_min
        self.axis_max = axis_max
    
    def play(self, start_frame, end_frame ):
        self.start_frame = start_frame
        self.end_frame = end_frame
                
        self.fig = plt.figure() 
        ax = plt3d.Axes3D(self.fig)
        ax.set_xlim(self.axis_min[0], self.axis_max[0])
        ax.set_ylim(self.axis_min[1], self.axis_max[1])
        ax.set_zlim(self.axis_min[2], self.axis_max[2])
        
        edge_count = len(self.edge_data)
        joint_dim = self.joint_data.shape[1]
        lines_data = [ np.empty([2,joint_dim]) for i in range(edge_count) ]

        self.lines = [ax.plot3D(line_data[0:1, 0], line_data[0:1, 1], line_data[0:1, 2], 'gray')[0] for line_data in lines_data]
        self.scatter = ax.scatter(np.zeros(self.joint_data.shape[0]), np.zeros(self.joint_data.shape[0]), np.zeros(self.joint_data.shape[0]))
        
        self.anim = animation.FuncAnimation(self.fig, self.progress_anim, frames=self.end_frame - self.start_frame, interval=20, blit=False) 
        self.fig.show()    

    def save(self, start_frame, end_frame, file_name ):
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        Writer = animation.writers['ffmpeg']
        self.writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
                
        self.fig = plt.figure() 
        ax = plt3d.Axes3D(self.fig)
        ax.set_xlim(self.axis_min[0], self.axis_max[0])
        ax.set_ylim(self.axis_min[1], self.axis_max[1])
        ax.set_zlim(self.axis_min[2], self.axis_max[2])
        
        edge_count = len(self.edge_data)
        joint_dim = self.joint_data.shape[1]
        lines_data = [ np.empty([2,joint_dim]) for i in range(edge_count) ]

        self.lines = [ax.plot3D(line_data[0:1, 0], line_data[0:1, 1], line_data[0:1, 2], 'gray')[0] for line_data in lines_data]
        self.scatter = ax.scatter(np.zeros(self.joint_data.shape[0]), np.zeros(self.joint_data.shape[0]), np.zeros(self.joint_data.shape[0]))
        
        self.anim = animation.FuncAnimation(self.fig, self.progress_anim, frames=self.end_frame - self.start_frame, interval=20, blit=False, repeat=False) 
        self.fig.show()    
        
        self.anim.save(file_name, writer=self.writer)

    def progress_anim(self, frame):
        self.progress_line_anim(frame)
        self.progress_scatter_anim(frame)

    def progress_line_anim(self, frame):
        frame = frame + self.start_frame

        #print("frame ", frame)

        for line, edge in zip(self.lines, self.edge_data):
            j0 = edge[0]
            j1 = edge[1]
            j0p = self.joint_data[frame,j0,:]
            j1p = self.joint_data[frame,j1,:]
            
            line_data = np.empty([2, 3])
            line_data[0] = j0p
            line_data[1] = j1p
            
            line.set_data(line_data[:, 0], line_data[:, 2])
            line.set_3d_properties(line_data[:, 1] * -1.0)

    def progress_scatter_anim(self, frame):
        frame = frame + self.start_frame
        
        #print(frame)

        pose = self.joint_data[frame, :, :]
        pose_x = [pose[i, 0] for i in range(pose.shape[0])]
        pose_y = [pose[i, 2] for i in range(pose.shape[0])]
        pose_z = [pose[i, 1] * -1.0 for i in range(pose.shape[0])]
    
        self.scatter._offsets3d = plt3d.art3d.juggle_axes(pose_x, pose_y, pose_z, 'z')