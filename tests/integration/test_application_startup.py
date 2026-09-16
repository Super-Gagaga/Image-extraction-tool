"""应用启动流程的集成测试。"""

from image_extraction_tool.main import create_application
from image_extraction_tool.ui.main_window import MainWindow


def test_main_window_can_be_constructed(qtbot) -> None:
    """验证主窗口能正常构造、显示，且初始状态下视图动作被禁用。"""
    create_application([])
    window = MainWindow()
    qtbot.addWidget(window)

    window.show()

    assert window.isVisible()
    assert window.windowTitle() == "智能抠图工具"
    assert window.centralWidget().objectName() == "canvasView"
    assert not window.fit_action.isEnabled()
