"""Управление курсором мыши: маппинг координат камеры на экран и сглаживание."""

import collections

import pyautogui

# Активная зона кадра (в долях от ширины/высоты), которую отображаем на весь экран.
# Края кадра оставляем как запас, чтобы рука не уходила за пределы обзора камеры.
DEFAULT_ROI = (0.2, 0.8, 0.2, 0.8)  # x1, x2, y1, y2

SMOOTHING_WINDOW = 5


class MouseController:
    """Двигает курсор по нормализованным координатам руки со сглаживанием."""

    def __init__(self, roi=DEFAULT_ROI, smoothing=SMOOTHING_WINDOW):
        self.screen_width, self.screen_height = pyautogui.size()
        self.roi_x1, self.roi_x2, self.roi_y1, self.roi_y2 = roi
        self.x_history = collections.deque(maxlen=smoothing)
        self.y_history = collections.deque(maxlen=smoothing)

    def reset(self):
        """Сбрасывает историю сглаживания (когда рука потеряна)."""
        self.x_history.clear()
        self.y_history.clear()

    def map_to_screen(self, x_norm, y_norm):
        """Переводит нормализованные координаты руки (0..1) в координаты экрана."""
        percent_x = (x_norm - self.roi_x1) / (self.roi_x2 - self.roi_x1)
        percent_y = (y_norm - self.roi_y1) / (self.roi_y2 - self.roi_y1)

        percent_x = max(0.0, min(1.0, percent_x))
        percent_y = max(0.0, min(1.0, percent_y))

        screen_x = int(percent_x * (self.screen_width - 1))
        screen_y = int(percent_y * (self.screen_height - 1))

        # Не даём курсору попасть ровно в (0, 0): там срабатывает failsafe pyautogui.
        return max(1, screen_x), max(1, screen_y)

    def move(self, x_norm, y_norm):
        """Плавно двигает курсор к координатам руки (скользящее среднее)."""
        target_x, target_y = self.map_to_screen(x_norm, y_norm)

        self.x_history.append(target_x)
        self.y_history.append(target_y)

        smooth_x = int(sum(self.x_history) / len(self.x_history))
        smooth_y = int(sum(self.y_history) / len(self.y_history))

        pyautogui.moveTo(smooth_x, smooth_y)

    def click(self):
        pyautogui.click()

    def double_click(self):
        pyautogui.doubleClick()
