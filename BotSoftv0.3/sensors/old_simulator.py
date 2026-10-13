class SimulatedSensors:
    def __init__(self): self.reset()
    def reset(self):
        self.values = {"touch_left":False,"touch_right":False,"thermal_hot":False,
                       "thermal_intensity":0.0,"motion_left":(0.0,0.0,0.0),
                       "motion_right":(0.0,0.0,0.0)}
    def set_touch(self,left,right):
        self.values["touch_left"]=bool(left); self.values["touch_right"]=bool(right)
    def set_thermal(self,hot,intensity):
        self.values["thermal_hot"]=bool(hot); self.values["thermal_intensity"]=max(0.0,min(1.0,float(intensity)))
    def set_motion(self,left,right):
        self.values["motion_left"]=tuple(map(float,left)); self.values["motion_right"]=tuple(map(float,right))
    def read(self): return dict(self.values)
