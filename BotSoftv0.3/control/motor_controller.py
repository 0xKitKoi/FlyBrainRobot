class MotorController:
    """Simulated four-joint pose planner only; no real hardware calls."""
    LIMITS=(20,160)
    POSES={"IDLE":(90,90,90,90),"LEFT_CONTACT":(120,65,90,90),
           "RIGHT_CONTACT":(90,90,60,115),"BOTH_CONTACT":(120,65,60,115),
           "THERMAL_STIMULUS":(105,100,75,80),"MOVING":(95,85,85,95),"ERROR":(90,90,90,90)}
    NAMES=("left_shoulder","left_elbow","right_shoulder","right_elbow")
    """
    def command_for(self,state):
        vals=self.POSES.get(state,self.POSES["IDLE"])
        lo,hi=self.LIMITS
        return {n:max(lo,min(hi,int(v))) for n,v in zip(self.NAMES,vals)}
    """
    # in motor_controller.py
    def command_for(self, state, steering=None, baseline=None):
        vals = self.POSES.get(state, self.POSES["IDLE"])
        lo, hi = self.LIMITS
        cmd = {n: max(lo, min(hi, int(v))) for n, v in zip(self.NAMES, vals)}
        if steering is not None and baseline is not None:
            cmd["servo_deg"] = 90 + max(-60, min(60, 150 * (steering - baseline)))
        else:
            cmd["servo_deg"] = 90
        return cmd
