"""颜色取样信息、容差和已选颜色列表面板。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from image_extraction_tool.domain.color_selection import RGBColor, SelectedColor


class ColorPanel(QDockWidget):
    """阶段 3 的可见参数面板，所有控件均连接真实行为。"""

    tolerance_changed = Signal(int)
    remove_requested = Signal(int)
    clear_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("颜色抠除", parent)
        self.setObjectName("colorPanel")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        content = QWidget(self)
        layout = QVBoxLayout(content)
        hover_layout = QHBoxLayout()
        self.hover_swatch = QLabel()
        self.hover_swatch.setObjectName("hoverColorSwatch")
        self.hover_swatch.setFixedSize(28, 28)
        self.hover_text = QLabel("将指针移到图片上查看原图颜色")
        self.hover_text.setObjectName("hoverColorText")
        self.hover_text.setWordWrap(True)
        hover_layout.addWidget(self.hover_swatch)
        hover_layout.addWidget(self.hover_text, 1)
        layout.addLayout(hover_layout)

        form = QFormLayout()
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setObjectName("colorToleranceSpinBox")
        self.tolerance_spin.setRange(0, 255)
        self.tolerance_spin.setValue(30)
        self.tolerance_spin.setToolTip("RGB 欧氏距离；0 表示仅匹配完全相同的颜色")
        form.addRow("颜色容差", self.tolerance_spin)
        layout.addLayout(form)

        layout.addWidget(QLabel("已选背景色"))
        self.color_list = QListWidget()
        self.color_list.setObjectName("selectedColorList")
        self.color_list.setMinimumWidth(220)
        layout.addWidget(self.color_list, 1)

        buttons = QHBoxLayout()
        self.remove_button = QPushButton("移除选中")
        self.remove_button.setObjectName("removeSelectedColorButton")
        self.clear_button = QPushButton("清空全部")
        self.clear_button.setObjectName("clearSelectedColorsButton")
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)
        self.setWidget(content)

        self.tolerance_spin.valueChanged.connect(self.tolerance_changed)
        self.remove_button.clicked.connect(self._request_remove)
        self.clear_button.clicked.connect(lambda _checked=False: self.clear_requested.emit())
        self.color_list.currentRowChanged.connect(self._update_button_states)
        self.set_document_enabled(False)
        self.set_hover_color(None)

    def set_document_enabled(self, enabled: bool) -> None:
        """启用或冻结会改变文档的颜色控件。"""
        self.tolerance_spin.setEnabled(enabled)
        self.color_list.setEnabled(enabled)
        self.clear_button.setEnabled(enabled and self.color_list.count() > 0)
        self.remove_button.setEnabled(enabled and self.color_list.currentRow() >= 0)

    def set_tolerance(self, value: int) -> None:
        """同步文档容差而不触发一次重复蒙版计算。"""
        blocked = self.tolerance_spin.blockSignals(True)
        self.tolerance_spin.setValue(value)
        self.tolerance_spin.blockSignals(blocked)

    def set_colors(self, colors: list[SelectedColor]) -> None:
        """用文档中的颜色状态重建可见列表。"""
        enabled = self.color_list.isEnabled()
        self.color_list.clear()
        for selected in colors:
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor(*selected.rgb))
            red, green, blue = selected.rgb
            item = QListWidgetItem(
                QIcon(pixmap),
                f"{selected.hex_value}   RGB({red}, {green}, {blue})",
            )
            item.setData(Qt.ItemDataRole.UserRole, selected.rgb)
            self.color_list.addItem(item)
        self.clear_button.setEnabled(enabled and bool(colors))
        self.remove_button.setEnabled(False)

    def set_hover_color(self, rgb: RGBColor | None) -> None:
        """显示原图悬停色块、RGB 和 HEX。"""
        if rgb is None:
            self.hover_swatch.setStyleSheet("border: 1px solid #888; background: transparent;")
            self.hover_swatch.setProperty("rgb", None)
            self.hover_text.setText("将指针移到图片上查看原图颜色")
            return
        red, green, blue = rgb
        hex_value = "#{:02X}{:02X}{:02X}".format(*rgb)
        self.hover_swatch.setStyleSheet(f"border: 1px solid #555; background: {hex_value};")
        self.hover_swatch.setProperty("rgb", rgb)
        self.hover_text.setText(f"RGB({red}, {green}, {blue})\n{hex_value}")

    def _request_remove(self) -> None:
        row = self.color_list.currentRow()
        if row >= 0:
            self.remove_requested.emit(row)

    def _update_button_states(self, row: int) -> None:
        self.remove_button.setEnabled(self.color_list.isEnabled() and row >= 0)
