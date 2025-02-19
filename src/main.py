import cv2
import numpy as np
import glob
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from rembg import remove
from PIL import Image
import pyautogui
from colorama import init, Fore, Style

init()
pyautogui.FAILSAFE = True 
pyautogui.PAUSE = 0.5

def get_mouse_position():
    """Capture current mouse position on Ctrl+C."""
    try:
        print(Fore.CYAN + "Move mouse to desired position and press Ctrl+C to capture..." + Style.RESET_ALL)
        while True:
            x, y = pyautogui.position()
            print(Fore.GREEN + f"Current position: ({x}, {y})" + Style.RESET_ALL, end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(Fore.YELLOW + f"\nPosition captured: ({x}, {y})" + Style.RESET_ALL)

def perform_click_action(count, pos1=(500, 500), pos2=(700, 500), pos3=(1225, 1080)):
    """Perform automated clicks based on object count."""
    try:
        if count >= 12:
            pyautogui.click(*pos1)
            print(Fore.YELLOW + f"Clicked at 'Large': {pos1}" + Style.RESET_ALL)
        else:
            pyautogui.click(*pos2) 
            print(Fore.GREEN + f"Clicked at 'Small': {pos2}" + Style.RESET_ALL)
        
        time.sleep(2)
        pyautogui.click(*pos3)
        print(Fore.YELLOW + f"Clicked at 'I want to challenge': {pos3}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Click action failed: {str(e)}" + Style.RESET_ALL)

def process_template(template_path, gray_image, image):
    """Process template matching with scale variations."""
    template = cv2.imread(template_path)
    if template is None:
        return 0
    
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = gray_template.shape[::-1]
    
    object_count = 0
    detected_boxes = []
    detected_centers = set()
    last_center = None
    
    scales = np.linspace(0.8, 2.0, 10)
    threshold = 0.77
    
    for scale in scales:
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        scaled_template = cv2.resize(gray_template, (scaled_w, scaled_h))
        
        result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        points = np.column_stack((locations[1], locations[0]))
        
        if len(points) == 0:
            continue
        
        for x1, y1 in points:
            x2, y2 = x1 + scaled_w, y1 + scaled_h
            center = (x1 + scaled_w//2, y1 + scaled_h//2)
            
            if last_center is not None and np.sqrt(np.sum((np.array(center) - np.array(last_center))**2)) < 20:
                continue
            
            if center not in detected_centers:
                new_box = np.array([x1, y1, x2, y2])
                
                if detected_boxes:
                    boxes = np.array(detected_boxes)
                    overlap = np.any(
                        (np.minimum(boxes[:, 2], new_box[2]) > np.maximum(boxes[:, 0], new_box[0])) &
                        (np.minimum(boxes[:, 3], new_box[3]) > np.maximum(boxes[:, 1], new_box[1]))
                    )
                    
                    if not overlap:
                        detected_boxes.append(new_box)
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        object_count += 1
                        detected_centers.add(center)
                        last_center = center
                else:
                    detected_boxes.append(new_box)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    object_count += 1
                    detected_centers.add(center)
                    last_center = center
    
    return object_count

def process_source_image(source_path):
    """Process a single source image and detect objects."""
    try:
        input_image = Image.open(source_path)
        output = remove(input_image, bgcolor=(255, 255, 255))
        output = output.convert('RGB')
        
        image = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        template_paths = glob.glob("templates/*.png")
        if not template_paths:
            print(Fore.RED + "No templates found" + Style.RESET_ALL)
            return 0
        
        total_objects = sum(process_template(path, gray_image, image) for path in template_paths)
        
        print(Fore.GREEN + f"Total objects found in {source_path}: {total_objects}" + Style.RESET_ALL)
        perform_click_action(total_objects, pos1=(1273, 924), pos2=(1559, 920), pos3=(1416, 1018))
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output_{os.path.basename(source_path)}")
        cv2.imwrite(output_path, image)
        
        return total_objects
    
    except Exception as e:
        print(Fore.RED + f"Error processing {source_path}: {str(e)}" + Style.RESET_ALL)
        return 0

def capture_and_crop(left, top, width, height, output_dir="res"):
    """Capture and crop a screenshot of specified region."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        screenshot = pyautogui.screenshot()
        cropped_image = screenshot.crop((left, top, left + width, top + height))
        
        filepath = os.path.join(output_dir, "screenshot.png")
        cropped_image.save(filepath)
        print(Fore.GREEN + f"Screenshot saved to: {filepath}" + Style.RESET_ALL)
        
        return filepath
        
    except Exception as e:
        print(Fore.RED + f"Error capturing screenshot: {str(e)}" + Style.RESET_ALL)
        return None

def capture_multiple_regions(regions):
    """Capture multiple screen regions."""
    return [filepath for region in regions 
            if (filepath := capture_and_crop(*region)) is not None]

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--get-pos', action='store_true', help='Get mouse position')
    args = parser.parse_args()

    if args.get_pos:
        get_mouse_position()
        return

    source_paths = glob.glob("res/*.png")
    if not source_paths:
        print(Fore.RED + "No source images found" + Style.RESET_ALL)
        raise ValueError("No source images found")
    
    print(Fore.CYAN + "Processing images..." + Style.RESET_ALL)
    start_time = time.time()
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_source_image, source_paths))
    
    total_objects = sum(results)
    processing_time = time.time() - start_time
    
    print(Fore.GREEN + f"Total objects found across all images: {total_objects}" + Style.RESET_ALL)
    print(Fore.CYAN + f"Total processing time: {processing_time:.2f} seconds" + Style.RESET_ALL)

if __name__ == "__main__":
    while True:
        time.sleep(3)
        capture_and_crop(990, 780, 150, 150)
        main()