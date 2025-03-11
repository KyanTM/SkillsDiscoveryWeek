import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import math

# Load assets
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Load plane image using OpenCV
PLANE_IMG = cv2.imread("plane.png", cv2.IMREAD_UNCHANGED)
if PLANE_IMG is None:
    print("ERROR: plane.png not found!")
    exit()
PLANE_IMG = cv2.resize(PLANE_IMG, (100, 100))
PLANE_IMG = cv2.rotate(PLANE_IMG, cv2.ROTATE_90_CLOCKWISE)
PLANE_IMG = cv2.cvtColor(PLANE_IMG, cv2.COLOR_BGRA2RGBA)
plane_surface = pygame.image.frombuffer(PLANE_IMG.tobytes(), PLANE_IMG.shape[1::-1], "RGBA")

# Load trash can image
TRASH_CAN_IMG = cv2.imread("bin.png", cv2.IMREAD_UNCHANGED)
if TRASH_CAN_IMG is None:
    print("ERROR: bin.png not found!")
    exit()
TRASH_CAN_IMG = cv2.resize(TRASH_CAN_IMG, (100, 500))  # Make trash cans larger
TRASH_CAN_IMG = cv2.cvtColor(TRASH_CAN_IMG, cv2.COLOR_BGRA2RGBA)
trash_can_surface = pygame.image.frombuffer(TRASH_CAN_IMG.tobytes(), TRASH_CAN_IMG.shape[1::-1], "RGBA")
trash_can_surface_flipped = pygame.transform.flip(trash_can_surface, False, True)  # Flipped trash can for top


# Load background images safely
def load_image_safe(filename, width, height):
    try:
        img = pygame.image.load(filename)
        return pygame.transform.scale(img, (width, height))
    except pygame.error as e:
        print(f"ERROR: Failed to load {filename} - {e}")
        # Create a fallback image instead of exiting
        fallback = pygame.Surface((width, height))
        fallback.fill((100, 100, 255))  # Fill with blue as fallback
        return fallback


# Modified to use full screen width for game (no camera display)
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600

STATIC_BG = load_image_safe("1.png", SCREEN_WIDTH, SCREEN_HEIGHT)
TOP_LAYER = load_image_safe("4.png", SCREEN_WIDTH, SCREEN_HEIGHT)
BOTTOM_LAYER = load_image_safe("5.png", SCREEN_WIDTH, SCREEN_HEIGHT)

# MediaPipe hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenCV setup for camera (still needed for hand tracking)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not found. Please check your connection and permissions.")
    exit()

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Hand-Controlled Plane Game')

# Background scrolling setup
bg_x = 0
scroll_speed = 5

# Smooth movement storage
position_history = []

# Trash can setup
trash_cans = []
trash_can_speed = 10  # Speed of horizontal movement
vertical_speed = 8  # Speed for vertical movement of bins
spawn_interval = 2000  # Milliseconds between spawns
last_spawn_time = pygame.time.get_ticks()

# Score setup
score = 0
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 36)
tiny_font = pygame.font.Font(None, 24)

# Game state
MENU = 0
PLAYING = 1
GAME_OVER = 2
game_state = MENU
high_score = 0

# Debug flag for collision visualization - now changeable by user
SHOW_COLLISION_BOXES = False

# Gesture detection storage
palm_detection_timer = 0
palm_detected = False
last_gesture_time = 0
gesture_cooldown = 1000  # Cooldown in milliseconds to prevent accidental triggers


def get_hand_info(frame, last_y):
    """
    Gets hand position and detects if the palm is facing the camera
    Returns: (y_position, is_palm_visible)
    """
    global position_history
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    y_pos = last_y
    is_palm_visible = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get hand position for plane control
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            y_pos = index_tip.y

            # Define crop limits
            crop_top = 0.2
            crop_bottom = 0.8
            cropped_y = (y_pos - crop_top) / (crop_bottom - crop_top)
            # Keep within bounds
            cropped_y = max(0, min(cropped_y, 1))
            # Scale to game height
            new_y = int(cropped_y * SCREEN_HEIGHT)

            # Store in history for smoothing
            position_history.append(new_y)
            if len(position_history) > 5:  # Keep history size of 5
                position_history.pop(0)

            # Compute smoothed position
            smoothed_y = sum(position_history) / len(position_history)
            # Keep the plane inside the screen
            smoothed_y = max(0, min(smoothed_y, SCREEN_HEIGHT - PLANE_IMG.shape[0]))

            # Detect if palm is visible (facing the camera)
            # Calculate vectors between landmarks
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            # If the middle finger base is closer to the camera than the wrist,
            # it likely means the palm is facing the camera
            if middle_mcp.z < wrist.z:
                is_palm_visible = True

            return int(smoothed_y), is_palm_visible

    return last_y, False


def spawn_trash_cans():
    # Spawn one bin at the top and one at the bottom of the screen
    # They will move towards the center to create a gap
    gap_height = 140  # Gap size (smaller gap for more challenge)

    # Range for the vertical position
    min_center = gap_height
    max_center = SCREEN_HEIGHT - gap_height

    # Divide the range into three zones for more variation
    zone_height = (max_center - min_center) // 3

    # Randomly choose a zone (top, middle, bottom)
    zone = random.randint(0, 2)

    # Determine the center of the opening within the chosen zone
    if zone == 0:  # Top zone
        gap_center = random.randint(min_center, min_center + zone_height)
    elif zone == 1:  # Middle zone
        gap_center = random.randint(min_center + zone_height, min_center + 2 * zone_height)
    else:  # Bottom zone
        gap_center = random.randint(min_center + 2 * zone_height, max_center)

    # Calculate final positions for top and bottom bins
    top_final_y = gap_center - gap_height / 2 - TRASH_CAN_IMG.shape[0]
    bottom_final_y = gap_center + gap_height / 2

    # Start positions (offscreen)
    top_start_y = -TRASH_CAN_IMG.shape[0]  # Start above the screen
    bottom_start_y = SCREEN_HEIGHT  # Start below the screen

    # [x position, current y, final y, is_top_bin]
    # Adding the is_top_bin flag to differentiate between top and bottom bins
    trash_cans.append([
        SCREEN_WIDTH,  # Start at the right edge of screen
        top_start_y,  # Start position
        top_final_y,  # Target position
        True  # This is a top bin
    ])

    trash_cans.append([
        SCREEN_WIDTH,  # Start at the right edge of screen
        bottom_start_y,  # Start position
        bottom_final_y,  # Target position
        False  # This is a bottom bin
    ])


def update_trash_cans():
    for trash_can in trash_cans:
        # Move horizontally
        trash_can[0] -= trash_can_speed

        # Move vertically towards final position
        current_y = trash_can[1]
        final_y = trash_can[2]

        if trash_can[3]:  # Top bin moves down
            if current_y < final_y:
                trash_can[1] = min(current_y + vertical_speed, final_y)
        else:  # Bottom bin moves up
            if current_y > final_y:
                trash_can[1] = max(current_y - vertical_speed, final_y)


def get_bin_hitbox(trash_can):
    """Function to consistently generate a hitbox for a trash bin"""
    is_top_bin = trash_can[3]
    current_y = trash_can[1]

    # Verbeterde hitbox voor vuilnisbakken (volle hoogte)
    return pygame.Rect(
        trash_can[0] + 15,  # Offset from left edge
        current_y + 10,  # Kleinere offset van top (was 25)
        TRASH_CAN_IMG.shape[1] - 30,  # Width (narrower than the image)
        TRASH_CAN_IMG.shape[0] - 20  # Grotere hoogte (was -50)
    )


def get_plane_hitboxes(plane_x, plane_y):
    """Create multiple hitboxes for the plane - including a hitbox for its nose and rear"""
    # Main body hitbox (slightly smaller than the image)
    body_rect = pygame.Rect(
        plane_x + 10,  # Kleinere X offset (was 25) zodat de achterkant meer hitbox heeft
        plane_y + 25,  # Y position with offset from top
        PLANE_IMG.shape[1] - 35,  # Width (smaller than the image)
        PLANE_IMG.shape[0] - 50  # Height (smaller than the image)
    )

    # Nose/front hitbox (to detect collisions at the front of the plane)
    nose_rect = pygame.Rect(
        plane_x + PLANE_IMG.shape[1] - 30,  # Right side of the plane
        plane_y + 40,  # Centered vertically
        30,  # Small width for the nose
        PLANE_IMG.shape[0] - 80  # Height (smaller than the body)
    )

    return [body_rect, nose_rect]


def check_collision(plane_hitboxes, trash_cans):
    """Improved collision detection that checks all plane hitboxes"""
    for trash_can in trash_cans:
        bin_rect = get_bin_hitbox(trash_can)

        # Check each plane hitbox against the bin
        for plane_rect in plane_hitboxes:
            if plane_rect.colliderect(bin_rect):
                return True

    return False


def draw_menu():
    # Draw the start menu
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))

    title = font.render("Hand-Controlled Plane Game", True, WHITE)
    start_text = small_font.render("Hold your hand up in front of the camera to start", True, WHITE)
    instruction = small_font.render("Move your hand up and down to control the plane", True, WHITE)
    hitbox_instruction = small_font.render("Show your palm to toggle hitboxes visibility", True, WHITE)

    screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, SCREEN_HEIGHT // 3 - title.get_height() // 2))
    screen.blit(start_text, (SCREEN_WIDTH // 2 - start_text.get_width() // 2, SCREEN_HEIGHT // 2))
    screen.blit(instruction, (SCREEN_WIDTH // 2 - instruction.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
    screen.blit(hitbox_instruction, (SCREEN_WIDTH // 2 - hitbox_instruction.get_width() // 2, SCREEN_HEIGHT // 2 + 100))


def draw_game_over():
    # Draw the game over screen
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))

    game_over_text = font.render("Game Over", True, WHITE)
    score_text = small_font.render(f"Your Score: {score}", True, WHITE)
    high_score_text = small_font.render(f"High Score: {high_score}", True, WHITE)
    restart_text = small_font.render("Show your palm to restart", True, WHITE)

    screen.blit(game_over_text, (
        SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 3 - game_over_text.get_height() // 2))
    screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2))
    screen.blit(high_score_text, (SCREEN_WIDTH // 2 - high_score_text.get_width() // 2, SCREEN_HEIGHT // 2 + 40))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 100))


def toggle_hitboxes():
    """Toggle the visibility of hitboxes"""
    global SHOW_COLLISION_BOXES
    SHOW_COLLISION_BOXES = not SHOW_COLLISION_BOXES


def reset_game():
    global score, trash_cans, last_spawn_time
    score = 0
    trash_cans = []
    last_spawn_time = pygame.time.get_ticks()
    position_history.clear()


def process_palm_gesture(is_palm_visible, elapsed_time):
    """Process palm gesture with a cooldown to prevent accidental triggers"""
    global palm_detection_timer, palm_detected, last_gesture_time

    current_time = pygame.time.get_ticks()

    # Check if enough time has passed since last gesture
    if current_time - last_gesture_time < gesture_cooldown:
        return False

    # If palm is visible, start counting
    if is_palm_visible:
        palm_detection_timer += elapsed_time
        # Need to hold palm visible for at least 1 second
        if palm_detection_timer > 1000 and not palm_detected:
            palm_detected = True
            palm_detection_timer = 0
            last_gesture_time = current_time
            return True
    else:
        palm_detection_timer = 0
        palm_detected = False

    return False


def main_loop():
    global bg_x, score, last_spawn_time, trash_cans, game_state, high_score, SHOW_COLLISION_BOXES
    plane_y = SCREEN_HEIGHT // 2  # Start in middle
    plane_x = 200  # Fixed x position for the plane
    running = True
    clock = pygame.time.Clock()
    start_timer = 0
    print("Game started!")

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Process camera for hand tracking
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Camera not found.")
            break

        # Get elapsed time for this frame
        elapsed_time = clock.get_time()

        # Process hand input and get both position and palm detection
        detected_y, is_palm_visible = get_hand_info(frame, plane_y)

        # Process palm gesture
        gesture_triggered = process_palm_gesture(is_palm_visible, elapsed_time)

        # State machine handling
        if game_state == MENU:
            # Draw background
            screen.blit(STATIC_BG, (0, 0))

            # Draw menu UI
            draw_menu()

            # Check if hand is detected to start game
            if len(position_history) >= 3:  # Ensure hand is stably detected
                start_timer += elapsed_time  # Add elapsed time
                if start_timer > 1500:  # 1.5 seconds of stable hand detection
                    reset_game()
                    game_state = PLAYING
                    start_timer = 0
            else:
                start_timer = 0  # Reset timer if hand disappeared

            # Toggle hitboxes if palm gesture detected
            if gesture_triggered:
                toggle_hitboxes()

        elif game_state == PLAYING:
            plane_y = detected_y

            # Toggle hitboxes if palm gesture detected
            if gesture_triggered:
                toggle_hitboxes()

            # Spawn trash cans at intervals
            current_time = pygame.time.get_ticks()
            if current_time - last_spawn_time > spawn_interval:
                spawn_trash_cans()
                last_spawn_time = current_time

            # Update trash can positions
            update_trash_cans()

            # Remove off-screen trash cans
            trash_cans = [trash_can for trash_can in trash_cans if trash_can[0] > -TRASH_CAN_IMG.shape[1]]

            # Get plane hitboxes (both body and nose)
            plane_hitboxes = get_plane_hitboxes(plane_x, plane_y)

            # Check for collisions with any part of the plane
            if check_collision(plane_hitboxes, trash_cans):
                if score > high_score:
                    high_score = score
                game_state = GAME_OVER
                start_timer = 0

            # Draw everything
            screen.blit(STATIC_BG, (0, 0))

            # Draw scrolling background
            bg_x = (bg_x - scroll_speed) % SCREEN_WIDTH
            screen.blit(TOP_LAYER, (bg_x - SCREEN_WIDTH, 0))
            screen.blit(TOP_LAYER, (bg_x, 0))
            screen.blit(BOTTOM_LAYER, (bg_x - SCREEN_WIDTH, 0))
            screen.blit(BOTTOM_LAYER, (bg_x, 0))

            # Draw the plane
            screen.blit(plane_surface, (plane_x, plane_y))

            # Draw trash cans
            for trash_can in trash_cans:
                if trash_can[3]:  # Top bin
                    screen.blit(trash_can_surface_flipped, (trash_can[0], trash_can[1]))
                else:  # Bottom bin
                    screen.blit(trash_can_surface, (trash_can[0], trash_can[1]))

            # Show hitboxes if enabled
            if SHOW_COLLISION_BOXES:
                # Draw plane hitboxes
                for hitbox in plane_hitboxes:
                    pygame.draw.rect(screen, GREEN, hitbox, 2)

                # Draw trash bin hitboxes
                for trash_can in trash_cans:
                    bin_rect = get_bin_hitbox(trash_can)
                    pygame.draw.rect(screen, RED, bin_rect, 2)

            # Update score
            score += 1
            score_text = font.render(str(score), True, (0, 0, 0))
            screen.blit(score_text, (SCREEN_WIDTH - 150, 50))

            # Show hitbox status
            hitbox_status = "HITBOXES: ON" if SHOW_COLLISION_BOXES else "HITBOXES: OFF"
            status_text = small_font.render(hitbox_status, True, WHITE)
            screen.blit(status_text, (50, 50))

            # Show palm gesture instruction
            hint_text = tiny_font.render("Show palm to toggle hitboxes", True, WHITE)
            screen.blit(hint_text, (50, 100))

            # Visualize palm detection progress if in progress
            if palm_detection_timer > 0:
                progress = min(palm_detection_timer / 1000, 1.0)  # 0.0 to 1.0
                pygame.draw.rect(screen, WHITE, (50, 130, 100, 10), 1)
                pygame.draw.rect(screen, GREEN, (50, 130, int(100 * progress), 10))

        elif game_state == GAME_OVER:
            # Draw basic scene in background
            screen.blit(STATIC_BG, (0, 0))

            # Draw game over screen
            draw_game_over()

            # Visualize palm detection progress if in progress
            if palm_detection_timer > 0:
                progress = min(palm_detection_timer / 1000, 1.0)  # 0.0 to 1.0
                bar_width = 200
                pygame.draw.rect(screen, WHITE,
                                 (SCREEN_WIDTH // 2 - bar_width // 2, SCREEN_HEIGHT // 2 + 150, bar_width, 20), 1)
                pygame.draw.rect(screen, GREEN, (
                SCREEN_WIDTH // 2 - bar_width // 2, SCREEN_HEIGHT // 2 + 150, int(bar_width * progress), 20))

            # Restart game if palm gesture detected
            if gesture_triggered:
                reset_game()
                game_state = PLAYING

        pygame.display.flip()
        clock.tick(60)

    print("Game Over! Your score:", score)
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


# Start the game
main_loop()