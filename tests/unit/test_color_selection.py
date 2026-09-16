"""颜色距离、容差蒙版和区域代表色的单元测试。"""

from PIL import Image

from image_extraction_tool.domain.color_selection import (
    SelectedColor,
    build_color_mask,
    representative_colors,
    rgb_distance_squared,
)


def test_rgb_distance_and_tolerance_boundary_are_inclusive() -> None:
    image = Image.new("RGBA", (3, 1))
    image.putdata(
        [
            (100, 100, 100, 255),
            (103, 104, 100, 255),  # 与目标色距离正好为 5
            (104, 104, 100, 255),  # 距离大于 5
        ]
    )
    selected = [SelectedColor((100, 100, 100))]

    mask = build_color_mask(image, selected, tolerance=5)

    assert rgb_distance_squared((100, 100, 100), (103, 104, 100)) == 25
    assert [mask.getpixel((x, 0)) for x in range(3)] == [0, 0, 255]


def test_multiple_colors_form_union_and_rebuild_does_not_accumulate() -> None:
    image = Image.new("RGBA", (3, 1))
    image.putdata([(10, 10, 10, 255), (20, 20, 20, 255), (40, 40, 40, 255)])
    colors = [SelectedColor((10, 10, 10)), SelectedColor((40, 40, 40))]

    strict = build_color_mask(image, colors, tolerance=0)
    broad = build_color_mask(image, colors, tolerance=18)
    strict_again = build_color_mask(image, colors, tolerance=0)

    assert [strict.getpixel((x, 0)) for x in range(3)] == [0, 255, 0]
    assert broad.getextrema() == (0, 0)
    assert strict_again.tobytes() == strict.tobytes()


def test_region_extracts_distinct_colors_and_ignores_transparent_pixels() -> None:
    image = Image.new("RGBA", (30, 10), (255, 0, 0, 255))
    for x in range(10, 20):
        for y in range(10):
            image.putpixel((x, y), (0, 255, 0, 255))
    for x in range(20, 30):
        for y in range(10):
            image.putpixel((x, y), (0, 0, 255, 0))

    colors = representative_colors(image, (0, 0, 30, 10), max_colors=8)
    values = {selected.rgb for selected in colors}

    assert values == {(255, 0, 0), (0, 255, 0)}
    assert len(values) == len(colors)


def test_duplicate_selected_color_is_rejected_by_document() -> None:
    from image_extraction_tool.domain.document import ImageDocument

    document = ImageDocument(Image.new("RGBA", (1, 1), "white"))
    selected = SelectedColor((1, 2, 3))

    assert document.add_selected_colors([selected, selected]) == [selected]
    assert document.add_selected_colors([selected]) == []
