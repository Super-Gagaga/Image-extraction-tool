"""``python -m image_extraction_tool`` 的模块入口。"""

from image_extraction_tool.main import main

# 直接以模块方式运行时启动应用，并把返回码交给解释器
raise SystemExit(main())
