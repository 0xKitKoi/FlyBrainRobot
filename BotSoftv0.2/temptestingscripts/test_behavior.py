import unittest
from control.behavior import BehaviorController
class TestBehavior(unittest.TestCase):
    def setUp(self):
        self.c=BehaviorController()
        self.s={"touch_left":False,"touch_right":False,"thermal_hot":False,
                "motion_left":(0,0,0),"motion_right":(0,0,0)}
    def test_states(self):
        for updates,expected in [({}, "IDLE"),({"touch_left":True},"LEFT_CONTACT"),
          ({"touch_right":True},"RIGHT_CONTACT"),
          ({"touch_left":True,"touch_right":True},"BOTH_CONTACT"),
          ({"thermal_hot":True},"THERMAL_STIMULUS")]:
            self.assertEqual(self.c.update({**self.s,**updates}),expected)
if __name__=="__main__": unittest.main()
