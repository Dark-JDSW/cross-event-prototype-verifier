"""Official CCPG inference transforms from FastPoseGait (pinned source).

Affine -> RescaleCenter -> HOD_MultiInput(coco) -> SkeletonInput.
Input confidence is preserved by geometry and excluded by the official HOD.
"""
import math
import numpy as np

class SkeletonInput(object):
    '''
    Transpose the input
    '''
    def __call__(self, data):
        # (T V C) -> (C T V)
        data = np.transpose(data, (2, 0, 1))
        return data[...,np.newaxis]

class HOD_MultiInput(object):
    '''
    Human-oriented descripter of GPGait
    Paper: https://arxiv.org/abs/2303.05234
    '''
    def __init__(self, joint_format='coco'):
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            self.angle_list = [
                np.array([0,1,2]),
                np.array([1,0,3]),
                np.array([2,4,0]),
                np.array([3,1]),
                np.array([4,2]),
                np.array([5,7,11]),
                np.array([6,8,12]),
                np.array([7,5,9]),
                np.array([8,10,6]),
                np.array([9,7]),
                np.array([10,8]),
                np.array([11,5,13]),
                np.array([12,6,14]),
                np.array([13,11,15]),
                np.array([14,12,16]),
                np.array([15,13]),
                np.array([16,14])
            ]
        elif joint_format == 'alphapose' or joint_format == 'openpose':
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            self.angle_list = [
                np.array([0,15,14]),
                np.array([15,0,17]),
                np.array([14,16,0]),
                np.array([17,15]),
                np.array([16,14]),
                np.array([5,6,11]),
                np.array([2,3,8]),
                np.array([6,3,8]),
                np.array([3,4,2]),
                np.array([7,6]),
                np.array([4,3]),
                np.array([11,5,12]),
                np.array([8,2,9]),
                np.array([12,11,13]),
                np.array([9,8,10]),
                np.array([13,12]),
                np.array([10,9])
            ]

    def cal_edge_angle(self,center,left):
        # generate center/left/right
        right = np.zeros_like(center)
        right[0,:] = center[0,:]
        right[1,:] = left[1,:]
        return self.cos_law(center,left,right)

    def cal_inner_angle(self,center,left,right):
        return self.cos_law(center,left,right)

    def cos_law(self,center,left,right):
        # adjacent_side_1
        side1 = np.sqrt(np.power(center[0,:]-left[0,:],2) + np.power(center[1,:]-left[1,:],2))
        # adjacent_side_2
        side2 = np.sqrt(np.power(center[0,:]-right[0,:],2) + np.power(center[1,:]-right[1,:],2))
        # opposite_side
        side3 = np.sqrt(np.power(left[0,:]-right[0,:],2) + np.power(left[1,:]-right[1,:],2))
        # cosine_law
        deno = side1*side2
        where_zero = np.where(deno==0)
        deno[where_zero] = 1
        cos = (side1*side1+side2*side2-side3*side3)/(2*deno)
        where_mo_one = np.where(cos > 1)
        where_le_one = np.where(cos < -1)
        cos[where_mo_one] = 1
        cos[where_le_one] = -1
        data_return = math.pi - np.arccos(cos)
        data_return[where_zero] = math.pi
        return data_return

    def __call__(self, data):
        data = np.transpose(data, (2, 0, 1))
        data = data[:2, :, :]

        C, T, V = data.shape
        data_new = np.zeros((3, C, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        # Bones
        for i in range(len(self.connect_joint)):
            data_new[1, :, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        # Angles
        for i in range(len(self.angle_list)):
            if len(self.angle_list[i]) == 3:
                data_new[2,:1,:,i] = self.cal_inner_angle(data[:,:,self.angle_list[i][0]],data[:,:,self.angle_list[i][1]],data[:,:,self.angle_list[i][2]])
            else:
                data_new[2,:1,:,i] = self.cal_edge_angle(data[:,:,self.angle_list[i][0]],data[:,:,self.angle_list[i][1]])

        data_new = np.nan_to_num(data_new)

        I, C, T, V = data_new.shape
        data_new = data_new.reshape(I*C, T, V)
        # (C T V) -> (T V C)
        data_new = np.transpose(data_new, (1, 2, 0))

        return data_new

class Affine(object):
    '''
    Make the spine of skeleton vertical to the ground
    Detailed description in paper: https://arxiv.org/abs/2303.05234

    The formula for counterclockwise rotation around a point (x0, y0):
    xx = x * cos - y * sin + x0 * (1-cos) + y0 * sin
    yy = x * sin + y * cos + y0 * (1-cos) - x0 * sin
    '''
    def __init__(self,fi = 0):
        self.fi = fi

    def __call__(self, data):
        kp = data
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]
        neck_x = (kp_x[:, 5]+kp_x[:, 6]) / 2
        neck_y = (kp_y[:, 5]+kp_y[:, 6]) / 2
        hip_x = (kp_x[:, 11] + kp_x[:, 12]) / 2
        hip_y = (kp_y[:, 11] + kp_y[:, 12]) / 2

        x_l = hip_x - neck_x
        y_l = hip_y - neck_y
        where_zero = np.where(y_l==0)
        y_l[where_zero] = 1
        theta = np.arctan(x_l / y_l)
        theta[where_zero] = 0.
        theta = np.expand_dims(theta, axis=-1)
        neck_x = np.expand_dims(neck_x, axis=-1)
        neck_y = np.expand_dims(neck_y, axis=-1)

        mask = np.where(np.abs(theta) > self.fi)[0]
        old_x = kp[mask, :, 0]
        old_y = kp[mask, :, 1]
        if len(mask) != 0:
            theta_rot = theta[mask]
            neck_x = neck_x[mask]
            neck_y = neck_y[mask]
            new_x = np.cos(theta_rot) * old_x - np.sin(theta_rot) * old_y + (1 - np.cos(theta_rot)) * neck_x + np.sin(theta_rot) * neck_y
            new_y = np.sin(theta_rot) * old_x + np.cos(theta_rot) * old_y + (1 - np.cos(theta_rot)) * neck_y - np.sin(theta_rot) * neck_x
            kp[mask, :, 0] = new_x
            kp[mask, :, 1] = new_y
        data = kp
        return data

class RescaleCenter(object):
    '''
    Normalize and align the skeletons
    Detailed description in paper: https://arxiv.org/abs/2303.05234
    '''
    def __init__(self,center='neck',scale=225):
        self.center = center
        self.scale = scale

    def __call__(self, data):

        kp = data # [T,17,3]

        if self.center=='None' or self.center ==None:
            return kp

        # Rescale
        kp_x = kp[...,0]
        kp_y = kp[...,1]
        min_y = np.min(kp_y, axis=1).reshape(-1, 1)
        max_y = np.max(kp_y, axis=1).reshape(-1, 1)
        old_h = max_y - min_y
        new_h = self.scale
        projection =  new_h / old_h
        kp_x *= projection
        kp_y *= projection

        # Center
        if self.center=='neck':
            offset_x = (kp_x[:, 5]+kp_x[:, 6])/2
            offset_y = (kp_y[:, 5]+kp_y[:, 6])/2
        elif self.center =='head':
            offset_x = (kp_x[:, 1]+kp_x[:, 2])/2
            offset_y = (kp_y[:, 1]+kp_y[:, 2])/2
        elif self.center =='hip':
            offset_x = (kp_x[:, 11]+kp_x[:, 12])/2
            offset_y = (kp_y[:, 11]+kp_y[:, 12])/2
        #offset
        kp_x -= offset_x.reshape(-1, 1)
        kp_y -= offset_y.reshape(-1, 1)
        data = kp
        return data
