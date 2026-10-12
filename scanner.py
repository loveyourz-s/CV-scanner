import cv2
import numpy as np
import time

def benchmark_document_scanner(image_path):
    print(" НАЧАЛО ОБРАБОТКИ ДОКУМЕНТА")
    total_start = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить файл: {image_path}")
    
    # 1. Предобработка
    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    t_blur = time.time() - t0
    print(f"[1/4] Предобработка (Blur/Gray): {t_blur*1000:.2f} мс")
    
    # 2. Бинаризация по Оцу и поиск контуров
    t0 = time.time()
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    document_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(c) > 5000:
            document_contour = approx
            break
    t_contours = time.time() - t0
    print(f"[2/4] Детекция границ и контуров: {t_contours*1000:.2f} мс")
    
    if document_contour is None:
        raise ValueError("Не удалось найти 4 угла документа на изображении!")
        
    # 3. Трансформация перспективы (A4)
    t0 = time.time()
    def order_points(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(document_contour.reshape(4, 2))
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))

    if maxHeight < maxWidth:
        maxWidth, maxHeight = maxHeight, maxWidth
    maxHeight = int(maxWidth * 1.414)

    dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    t_warp = time.time() - t0
    print(f"[3/4] Коррекция перспективы (Warp): {t_warp*1000:.2f} мс")

    # 4. Финальная адаптивная бинаризация
    t0 = time.time()
    gray_scan = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    final_scan = cv2.adaptiveThreshold(gray_scan, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11)
    t_thresh = time.time() - t0
    print(f"[4/4] Адаптивная бинаризация (Scan Effect): {t_thresh*1000:.2f} мс")
    
    total_time = time.time() - total_start
    print(f" ОБЩЕЕ ВРЕМЯ ОБРАБОТКИ: {total_time*1000:.2f} мс")
    
    return warped, final_scan

if __name__ == "__main__":
    # Пример вызова
    # warped, scanned = benchmark_document_scanner('IMG_0259.JPG')
    pass
