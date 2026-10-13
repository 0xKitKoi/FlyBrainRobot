class BehaviorController:
    def update(self,s):
        if s.get("brain_error"): return "ERROR"
        if s.get("touch_left") and s.get("touch_right"): return "BOTH_CONTACT"
        if s.get("touch_left"): return "LEFT_CONTACT"
        if s.get("touch_right"): return "RIGHT_CONTACT"
        if s.get("thermal_hot"): return "THERMAL_STIMULUS"
        motion=(*s.get("motion_left",(0,0,0)),*s.get("motion_right",(0,0,0)))
        if any(abs(v)>0.5 for v in motion): return "MOVING"
        return "IDLE"
