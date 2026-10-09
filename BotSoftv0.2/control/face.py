def render_status(s,state,pose):
    faces={"IDLE":"( o.o )","LEFT_CONTACT":"( >.< )","RIGHT_CONTACT":"( >.< )",
           "BOTH_CONTACT":"( x.x )","THERMAL_STIMULUS":"( *.* )","MOVING":"( ^.^ )","ERROR":"( !!! )"}
    return (f"\n+--------------------------------------+\n| {faces.get(state,'( o.o )'):<36} |\n"
            f"| STATE: {state:<29} |\n"
            f"| TOUCH L/R: {str(s.get('touch_left')):<5} / {str(s.get('touch_right')):<5}       |\n"
            f"| THERMAL: {s.get('thermal_intensity',0):.2f} hot={str(s.get('thermal_hot',False)):<5}             |\n"
            f"| BRAIN STEERING: {s.get('brain_steering','n/a')!s:<21} |\n"
            f"| POSE: {pose!s:<30} |\n+--------------------------------------+")
