from pymavlink import mavutil

# Connect to Pixhawk over USB
mav = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
mav.wait_heartbeat()
print("Connected to system %u" % mav.target_system)

# Send servo command: channel 1, 1500 µs PWM
mav.mav.command_long_send(
    mav.target_system,
    mav.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
    0,      # confirmation
    1,      # servo channel (e.g., SERVO1)
    1500,   # PWM in microseconds (1000–2000 typical)
    0, 0, 0, 0, 0
)

print("Sent 1500 µs PWM to SERVO1")