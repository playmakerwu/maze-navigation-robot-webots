# -*- coding: utf-8 -*-
"""student_controller controller."""

import math
import numpy as np
import random

# --- Helper Functions (Degrees) ---

def normalize_angle(angle_deg):
    """Normalize an angle in degrees to the range [-180, 180)."""
    while angle_deg >= 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg

def angle_diff(angle1_deg, angle2_deg):
    """Calculate the smallest difference between two angles in degrees (result in [-180, 180])."""
    diff = normalize_angle(angle1_deg - angle2_deg)
    return diff

def gaussian_likelihood(x, mu, sigma):
    """Calculate the likelihood of x under a Gaussian distribution with mean mu and std dev sigma."""
    if sigma <= 1e-9:
        return 1.0 if abs(x - mu) < 1e-9 else 0.0
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    denominator = sigma * math.sqrt(2 * math.pi)
    if denominator == 0: return 0.0
    return (1.0 / denominator) * math.exp(exponent)


# --- Particle Filter Class (Using Degrees Internally for Robot Pose) ---
# (Particle Filter class remains largely the same)
class ParticleFilter:
    def __init__(self, landmarks, num_particles=100, motion_noise_std_dev=None, sensor_noise_std_dev=None, initial_pose=None):
        self.num_particles = num_particles
        self.landmarks = landmarks
        self.motion_noise = motion_noise_std_dev if motion_noise_std_dev else [0.05, 2.0] # m, degrees
        self.sensor_noise = sensor_noise_std_dev if sensor_noise_std_dev else [0.2, 5.0]   # m, degrees
        self.fov_deg = 90.0

        self.particles = []
        if initial_pose:
            init_x, init_y, init_theta_norm_deg = initial_pose
            init_theta_deg = normalize_angle(init_theta_norm_deg)
            init_std_dev = [0.2, 0.2, 5.0] # x, y, theta_deg
            for _ in range(self.num_particles):
                x = random.gauss(init_x, init_std_dev[0])
                y = random.gauss(init_y, init_std_dev[1])
                theta_deg = normalize_angle(random.gauss(init_theta_deg, init_std_dev[2]))
                self.particles.append([x, y, theta_deg, 1.0 / self.num_particles])
        else:
            for _ in range(self.num_particles):
                x = random.uniform(-5.0, 5.0)
                y = random.uniform(-3.5, 3.5)
                theta_deg = random.uniform(-180.0, 180.0)
                self.particles.append([x, y, theta_deg, 1.0 / self.num_particles])

    def predict(self, odometry):
        delta_forward, delta_rotation_rad = odometry
        delta_rotation_deg = math.degrees(delta_rotation_rad)

        for i in range(self.num_particles):
            particle = self.particles[i]
            noisy_delta_forward = random.gauss(delta_forward, self.motion_noise[0] * abs(delta_forward) + 0.01)
            noisy_delta_rotation_deg = random.gauss(delta_rotation_deg, self.motion_noise[1] * abs(delta_rotation_deg) + 0.5)
            current_theta_rad = math.radians(particle[2])
            delta_x = noisy_delta_forward * math.cos(current_theta_rad)
            delta_y = noisy_delta_forward * math.sin(current_theta_rad)
            particle[0] += delta_x
            particle[1] += delta_y
            particle[2] = normalize_angle(particle[2] + noisy_delta_rotation_deg)

    def update(self, sensor_observations):
        total_weight = 0.0
        for i in range(self.num_particles):
            particle = self.particles[i]
            px, py, ptheta_deg, _ = particle
            likelihood = 1.0
            landmark_obs_types = {
                'goal': sensor_observations.get('goal', []),
                'center_circle': [sensor_observations.get('center_circle')] if sensor_observations.get('center_circle') else [],
                'penalty_cross': sensor_observations.get('penalty_cross', []),
                'corners': sensor_observations.get('corners', [])
            }
            landmark_map_keys = {
                'goal': ['goal1', 'goal2'], 'center_circle': ['center'],
                'penalty_cross': ['cross1', 'cross2'],
                'corners': ['corner1', 'corner2', 'corner3', 'corner4']
            }
            num_valid_observations = 0
            for obs_type, observations in landmark_obs_types.items():
                if not observations: continue
                map_landmark_keys = landmark_map_keys[obs_type]
                for obs_dist, obs_angle_rad in observations:
                    if obs_dist is None or obs_dist > 10.0: continue
                    num_valid_observations += 1
                    obs_angle_deg = math.degrees(obs_angle_rad)
                    max_landmark_likelihood = 0.0
                    for landmark_key in map_landmark_keys:
                        lx, ly = self.landmarks[landmark_key]
                        dx, dy = lx - px, ly - py
                        expected_dist = math.sqrt(dx**2 + dy**2)
                        abs_angle_to_landmark_deg = normalize_angle(math.degrees(math.atan2(dy, dx)))
                        expected_rel_angle_deg = angle_diff(abs_angle_to_landmark_deg, ptheta_deg)
                        if abs(expected_rel_angle_deg) <= self.fov_deg / 2.0:
                            dist_likelihood = gaussian_likelihood(obs_dist, expected_dist, self.sensor_noise[0])
                            angle_error_deg = angle_diff(obs_angle_deg, expected_rel_angle_deg)
                            angle_likelihood = gaussian_likelihood(angle_error_deg, 0, self.sensor_noise[1])
                            current_landmark_likelihood = dist_likelihood * angle_likelihood
                            if current_landmark_likelihood > max_landmark_likelihood:
                                max_landmark_likelihood = current_landmark_likelihood
                    likelihood *= (max_landmark_likelihood + 1e-9)
            particle[3] *= likelihood
            total_weight += particle[3]

        if total_weight > 1e-9:
            norm_factor = 1.0 / total_weight
            for i in range(self.num_particles):
                self.particles[i][3] *= norm_factor
        else:
            uniform_weight = 1.0 / self.num_particles
            for i in range(self.num_particles):
                self.particles[i][3] = uniform_weight

    def resample(self):
        new_particles = []
        N = self.num_particles
        weights = [p[3] for p in self.particles]
        sum_weights = sum(weights)
        if sum_weights < 1e-9 or N == 0:
             uniform_weight = 1.0 / N if N > 0 else 0
             self.particles = [[p[0], p[1], p[2], uniform_weight] for p in self.particles]
             return
        if abs(sum_weights - 1.0) > 1e-6:
             norm_weights = [w / sum_weights for w in weights]
        else:
             norm_weights = weights
        M = N
        r = random.uniform(0, 1.0 / M)
        c = norm_weights[0]
        i = 0
        for m in range(M):
            U = r + m * (1.0 / M)
            while U > c:
                i = (i + 1) % N
                if i >= len(norm_weights): i = 0
                c += norm_weights[i]
            new_particle = list(self.particles[i])
            new_particle[3] = 1.0 / N
            new_particles.append(new_particle)
        self.particles = new_particles

    def estimate_pose(self):
        mean_x, mean_y, mean_vx, mean_vy = 0.0, 0.0, 0.0, 0.0
        total_weight = sum(p[3] for p in self.particles)
        if total_weight < 1e-9 or not self.particles:
            return (0, 0, 0) # Default pose
        for x, y, theta_deg, weight in self.particles:
            w_norm = weight / total_weight if total_weight > 0 else 1.0 / self.num_particles
            mean_x += w_norm * x
            mean_y += w_norm * y
            theta_rad = math.radians(theta_deg)
            mean_vx += w_norm * math.cos(theta_rad)
            mean_vy += w_norm * math.sin(theta_rad)
        mean_theta_rad = math.atan2(mean_vy, mean_vx)
        mean_theta_deg = normalize_angle(math.degrees(mean_theta_rad))
        return (mean_x, mean_y, mean_theta_deg)

# --- Student Controller Class ---

class StudentController:
    def __init__(self):
        self.landmarks = {
            'center': (0.0, 0.0), 'goal1': (4.5, 0.0), 'goal2': (-4.5, 0.0),
            'cross1': (3.25, 0.0), 'cross2': (-3.25, 0.0),
            'corner1': (-4.5, 3.0), 'corner2': (-4.5, -3.0),
            'corner3': (4.5, 3.0), 'corner4': (4.5, -3.0)
        }
        self.initial_pose = (-1.0, 0.0, 0.0)
        self.target_goal_pos = self.landmarks['goal1']

        pf_motion_noise = [0.1, 3.0]
        pf_sensor_noise = [0.25, 6.0]
        self.pf = ParticleFilter(self.landmarks,
                                 num_particles=300,
                                 initial_pose=self.initial_pose,
                                 motion_noise_std_dev=pf_motion_noise,
                                 sensor_noise_std_dev=pf_sensor_noise)

        self.estimated_pose = self.initial_pose
        self.estimated_ball_pos = None
        self.ball_estimate_smoothing = 0.4
        self.time_ball_last_seen = -1000.0
        self.current_sim_time = 0.0

        # --- State Machine ---
        self.state = "SEARCHING_FOR_BALL"
        self.alignment_target_pos = None         # Target (x, y) for alignment maneuver
        self.recorded_ball_pos_for_turn = None # Ball position recorded when calculating alignment target

        self.resample_counter = 0

        # --- Physical Parameters ---
        self.robot_radius = 0.18
        self.ball_radius = 0.065
        # --- Control Parameters ---
        self.max_speed = 6.0 # Consistent with previous user versions
        self.alignment_clearance = 0.05 # Reduced clearance from user's last version
        self.nav_tolerance = 0.10 # Reduced tolerance for reaching align point
        self.turn_tolerance_deg = 5.0 # Tolerance for facing ball direction
        self.push_speed = self.max_speed

        # --- Variable to store last seen ball angle ---
        # NOTE: This variable was added for smarter search but is NOT used in the
        # SEARCHING state logic of THIS specific code version provided by the user previously.
        # It IS updated when the ball is seen, however.
        self.last_known_ball_rel_angle_deg = 0.0


    def navigate_to_point(self, target_pos):
        """ Helper function to generate controls to move towards a target (x, y) point. """
        robot_x, robot_y, robot_theta_deg = self.estimated_pose
        target_x, target_y = target_pos
        dx = target_x - robot_x; dy = target_y - robot_y
        dist_to_target = math.sqrt(dx**2 + dy**2)
        # Handle potential case where dist_to_target is exactly zero
        if dist_to_target < 1e-6:
             return 0.0, 0.0, 0.0 # No movement needed

        abs_angle_to_target_rad = math.atan2(dy, dx)
        abs_angle_to_target_deg = normalize_angle(math.degrees(abs_angle_to_target_rad))
        turn_angle_deg = angle_diff(abs_angle_to_target_deg, robot_theta_deg)
        turn_gain = 2.0; turn_speed = turn_gain * (turn_angle_deg / 45.0)
        turn_speed = np.clip(turn_speed, -self.max_speed * 0.7, self.max_speed * 0.7)
        base_speed = self.max_speed * 0.8 # Note: using 0.8 here from user code
        speed_reduction_factor = max(0.1, 1.0 - (abs(turn_angle_deg) / 90.0))
        # Use dist_to_target in division, ensure it's not zero
        approach_speed = base_speed * speed_reduction_factor * min(1.0, dist_to_target / max(1.0, dist_to_target)) # Avoid division by zero if dist is exactly 1? Better use fixed divisor or clip.
        approach_speed = base_speed * speed_reduction_factor * min(1.0, dist_to_target / 1.0) # Reverting to simpler scaling
        approach_speed = np.clip(approach_speed, 0, base_speed)
        left_motor = np.clip(approach_speed - turn_speed, -self.max_speed, self.max_speed)
        right_motor = np.clip(approach_speed + turn_speed, -self.max_speed, self.max_speed)
        return left_motor, right_motor, dist_to_target

    def turn_towards_point(self, target_pos):
        """ Helper function to generate controls to turn towards a target (x, y) point. """
        robot_x, robot_y, robot_theta_deg = self.estimated_pose
        target_x, target_y = target_pos
        dx = target_x - robot_x; dy = target_y - robot_y
        # Handle potential case where target is identical to robot pose
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0, 0.0, 0.0 # No turning needed

        abs_angle_to_target_rad = math.atan2(dy, dx)
        abs_angle_to_target_deg = normalize_angle(math.degrees(abs_angle_to_target_rad))
        turn_angle_deg = angle_diff(abs_angle_to_target_deg, robot_theta_deg)
        turn_gain = 1.5; turn_speed = turn_gain * (turn_angle_deg / 45.0)
        min_abs_turn_speed = 0.5
        if abs(turn_speed) < min_abs_turn_speed and abs(turn_angle_deg) > self.turn_tolerance_deg * 0.5:
             turn_speed = np.sign(turn_speed) * min_abs_turn_speed
        turn_speed = np.clip(turn_speed, -self.max_speed * 0.6, self.max_speed * 0.6)
        left_motor = -turn_speed; right_motor = turn_speed
        return left_motor, right_motor, turn_angle_deg


    def step(self, sensors):
        """
        Compute robot control using the decoupled navigation/observation logic and smarter search.
        """
        # --- 1. Update Simulation Time ---
        self.current_sim_time += 0.032

        # --- 2. Localization Step (Robot Pose) ---
        odometry = sensors['odometry']
        self.pf.predict(odometry); self.pf.update(sensors)
        self.resample_counter += 1
        if self.resample_counter >= 5: self.pf.resample(); self.resample_counter = 0
        self.estimated_pose = self.pf.estimate_pose()
        robot_x, robot_y, robot_theta_deg = self.estimated_pose
        robot_theta_rad = math.radians(robot_theta_deg)

        # --- 3. Ball Position Estimation Step ---
        ball_obs = sensors.get('ball')
        ball_currently_visible = False
        current_ball_rel_angle_deg = None # Initialize for safety

        if ball_obs:
            ball_dist, ball_rel_angle_rad = ball_obs
            ball_rel_angle_deg = math.degrees(ball_rel_angle_rad)
            if abs(ball_rel_angle_deg) <= (self.pf.fov_deg / 2.0 + 5.0):
                ball_currently_visible = True
                self.time_ball_last_seen = self.current_sim_time
                # Store the *current* relative angle before smoothing
                current_ball_rel_angle_deg = ball_rel_angle_deg
                # Update last known angle using the current observation
                self.last_known_ball_rel_angle_deg = current_ball_rel_angle_deg # Update here

                # Update smoothed estimate
                global_ball_angle_rad = robot_theta_rad + ball_rel_angle_rad
                current_calc_ball_x = robot_x + ball_dist * math.cos(global_ball_angle_rad)
                current_calc_ball_y = robot_y + ball_dist * math.sin(global_ball_angle_rad)
                current_calculated_ball_pos = (current_calc_ball_x, current_calc_ball_y)
                if self.estimated_ball_pos is None: self.estimated_ball_pos = current_calculated_ball_pos
                else:
                    est_x, est_y = self.estimated_ball_pos; alpha = self.ball_estimate_smoothing
                    new_est_x = alpha * current_calc_ball_x + (1 - alpha) * est_x
                    new_est_y = alpha * current_calc_ball_y + (1 - alpha) * est_y
                    self.estimated_ball_pos = (new_est_x, new_est_y)

        ball_estimate_reliable = False
        if self.estimated_ball_pos is not None:
            if (self.current_sim_time - self.time_ball_last_seen) < 1.0:
                 ball_estimate_reliable = True

        # --- 4. State Machine Logic ---
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}
        next_state = self.state
        current_target_point = None

        # ----- State: SEARCHING_FOR_BALL -----
        if self.state == "SEARCHING_FOR_BALL":
            # --- Using the FIXED rotation from the user's version ---
            # This will always turn left. The 'smarter search' is NOT implemented here
            # based on the request to debug the user's specific code version first.
            if self.last_known_ball_rel_angle_deg > 0:
                # Turn right if the last known ball angle is positive
                control_dict["left_motor"] = self.max_speed * 0.5
                control_dict["right_motor"] = -self.max_speed * 0.5
            else:
                control_dict["left_motor"] = -self.max_speed * 0.5
                control_dict["right_motor"] = self.max_speed * 0.5
            # --- End fixed rotation block ---
            current_target_point = "Searching"
            if ball_currently_visible:
                if ball_estimate_reliable:
                     next_state = "CALCULATING_ALIGN_TARGETS"

        # ----- State: CALCULATING_ALIGN_TARGETS -----
        elif self.state == "CALCULATING_ALIGN_TARGETS":
            current_target_point = "Calculating Targets"
            if not ball_estimate_reliable:
                next_state = "SEARCHING_FOR_BALL"; self.alignment_target_pos = None; self.recorded_ball_pos_for_turn = None
            else:
                self.recorded_ball_pos_for_turn = self.estimated_ball_pos
                ball_x, ball_y = self.recorded_ball_pos_for_turn
                goal_x, goal_y = self.target_goal_pos
                goal_dx = goal_x - ball_x; goal_dy = goal_y - ball_y
                dist_ball_to_goal = math.sqrt(goal_dx**2 + goal_dy**2)

                # --- MODIFICATION: Robust check for zero distance ---
                epsilon = 1e-6 # Define a small epsilon
                if dist_ball_to_goal <= epsilon:
                # --- END MODIFICATION ---
                    # Ball is at or extremely close to the goal
                    next_state = "PUSHING_BALL"
                    self.alignment_target_pos = None
                    self.recorded_ball_pos_for_turn = None
                else:
                    # Proceed with calculation only if distance is clearly non-zero
                    ux_goal_to_ball = (ball_x - goal_x) / dist_ball_to_goal
                    uy_goal_to_ball = (ball_y - goal_y) / dist_ball_to_goal
                    total_offset = self.alignment_clearance + self.ball_radius + self.robot_radius
                    align_x = ball_x + ux_goal_to_ball * total_offset
                    align_y = ball_y + uy_goal_to_ball * total_offset
                    self.alignment_target_pos = (align_x, align_y)
                    next_state = "NAVIGATING_TO_ALIGN_POINT"

        # ----- State: NAVIGATING_TO_ALIGN_POINT -----
        elif self.state == "NAVIGATING_TO_ALIGN_POINT":
            if self.alignment_target_pos is None:
                next_state = "SEARCHING_FOR_BALL"; current_target_point = "Error - No Align Target"; self.recorded_ball_pos_for_turn = None
            else:
                current_target_point = self.alignment_target_pos
                l_motor, r_motor, dist_to_target = self.navigate_to_point(self.alignment_target_pos)
                # Check if navigate_to_point returned valid numbers
                if not (math.isnan(l_motor) or math.isnan(r_motor)):
                    control_dict["left_motor"] = l_motor; control_dict["right_motor"] = r_motor
                else:
                    # Fallback if NaN occurs during navigation (e.g., target too close)
                    print("Warning: NaN detected in navigation. Reverting to SEARCH.")
                    next_state = "SEARCHING_FOR_BALL"; self.alignment_target_pos = None; self.recorded_ball_pos_for_turn = None

                if dist_to_target < self.nav_tolerance:
                    next_state = "TURNING_TOWARDS_BALL_POS"; self.alignment_target_pos = None

        # ----- State: TURNING_TOWARDS_BALL_POS -----
        elif self.state == "TURNING_TOWARDS_BALL_POS":
            if self.recorded_ball_pos_for_turn is None:
                 next_state = "SEARCHING_FOR_BALL"; current_target_point = "Error - No Recorded Ball Pos"
            else:
                current_target_point = self.recorded_ball_pos_for_turn
                l_motor, r_motor, turn_angle_deg = self.turn_towards_point(self.recorded_ball_pos_for_turn)
                 # Check if turn_towards_point returned valid numbers
                if not (math.isnan(l_motor) or math.isnan(r_motor)):
                    control_dict["left_motor"] = l_motor; control_dict["right_motor"] = r_motor
                else:
                    # Fallback if NaN occurs during turning
                    print("Warning: NaN detected in turning. Reverting to SEARCH.")
                    next_state = "SEARCHING_FOR_BALL"; self.recorded_ball_pos_for_turn = None

                if abs(turn_angle_deg) < self.turn_tolerance_deg:
                    if ball_currently_visible: next_state = "PUSHING_BALL"
                    else: next_state = "SEARCHING_FOR_BALL"
                    self.recorded_ball_pos_for_turn = None

        # ----- State: PUSHING_BALL -----
        elif self.state == "PUSHING_BALL":
            current_target_point = self.target_goal_pos
            push_ball_reliable = self.estimated_ball_pos is not None and (self.current_sim_time - self.time_ball_last_seen) < 1.5
            if not push_ball_reliable:
                next_state = "SEARCHING_FOR_BALL"; self.estimated_ball_pos = None
            else:
                goal_x, goal_y = self.target_goal_pos; ball_x, ball_y = self.estimated_ball_pos
                ball_dx = ball_x - robot_x; ball_dy = ball_y - robot_y; dist_robot_to_ball = math.sqrt(ball_dx**2 + ball_dy**2) # Typo corrected: dy**2
                goal_dx = goal_x - robot_x; goal_dy = goal_y - robot_y
                abs_angle_to_goal_rad = math.atan2(goal_dy, goal_dx)
                abs_angle_to_goal_deg = normalize_angle(math.degrees(abs_angle_to_goal_rad))
                turn_angle_to_goal = angle_diff(abs_angle_to_goal_deg, robot_theta_deg)
                target_turn_angle = turn_angle_to_goal

                if ball_currently_visible and dist_robot_to_ball < (self.robot_radius + self.ball_radius + 0.1) and current_ball_rel_angle_deg is not None:
                     target_turn_angle = 0.7 * turn_angle_to_goal + 0.3 * current_ball_rel_angle_deg
                     self.last_known_ball_rel_angle_deg = current_ball_rel_angle_deg

                turn_gain = 1.0; turn_speed = turn_gain * (target_turn_angle / 30.0)
                turn_speed = np.clip(turn_speed, -self.max_speed * 0.4, self.max_speed * 0.4)

                forward_speed = self.push_speed
                # Check for NaN before assigning
                if not (math.isnan(forward_speed) or math.isnan(turn_speed)):
                    control_dict["left_motor"] = np.clip(forward_speed - turn_speed, -self.max_speed, self.max_speed)
                    control_dict["right_motor"] = np.clip(forward_speed + turn_speed, -self.max_speed, self.max_speed)
                else:
                    print(f"Warning: NaN detected in PUSHING calculation (fw:{forward_speed}, ts:{turn_speed}). Setting motors to 0.")
                    control_dict["left_motor"] = 0.0
                    control_dict["right_motor"] = 0.0
                    next_state = "SEARCHING_FOR_BALL" # Revert to search if calculation failed

                if dist_robot_to_ball < (self.robot_radius + self.ball_radius + 0.2):
                    # Check for potential division by zero before atan2 if needed
                    if abs(ball_dx) > 1e-6 or abs(ball_dy) > 1e-6:
                         abs_angle_to_ball_rad = math.atan2(ball_dy, ball_dx)
                         rel_angle_ball_deg = angle_diff(math.degrees(abs_angle_to_ball_rad), robot_theta_deg)
                         if abs(rel_angle_ball_deg) > 135:
                             next_state = "SEARCHING_FOR_BALL"; self.estimated_ball_pos = None
                    # else: ball is extremely close to robot center, angle is ill-defined, maybe do nothing?


        # Update state for the next iteration
        self.state = next_state

        # --- 5. Print ONLY Ball Position and Target Point ---
        # Print statements commented out as per user request in previous turns
        '''
        if self.estimated_ball_pos:
            print(f"Ball Position: ({self.estimated_ball_pos[0]:.2f}, {self.estimated_ball_pos[1]:.2f})")
        else:
            print("Ball Position: None")
        if isinstance(current_target_point, str):
             print(f"Target Point: {current_target_point}")
        elif current_target_point is not None:
             print(f"Target Point: ({current_target_point[0]:.2f}, {current_target_point[1]:.2f})")
        else:
             print("Target Point: None (State Unknown)")
        '''

        # --- 6. Final Sanity Check and Return Control Dictionary ---
        # Ensure values are not NaN before returning
        if math.isnan(control_dict["left_motor"]):
             print("Error: left_motor is NaN before return! Setting to 0.")
             control_dict["left_motor"] = 0.0
        if math.isnan(control_dict["right_motor"]):
             print("Error: right_motor is NaN before return! Setting to 0.")
             control_dict["right_motor"] = 0.0

        control_dict["left_motor"] = np.clip(control_dict["left_motor"], -self.max_speed, self.max_speed)
        control_dict["right_motor"] = np.clip(control_dict["right_motor"], -self.max_speed, self.max_speed)
        return control_dict