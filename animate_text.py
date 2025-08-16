#!/usr/bin/env python3
"""
Production-ready handwriting animation with accurate text shaping and stroke ordering.

Key improvements:
- Proper curve flattening with adaptive subdivision
- Accurate glyph positioning and metrics handling
- Smart stroke ordering (outside-to-inside, top-to-bottom)
- Robust component glyph handling
- Better baseline and positioning calculations
- Optimized rendering with proper antialiasing
- Error handling and fallbacks

Dependencies: fonttools, uharfbuzz, numpy, matplotlib, scipy
"""
from __future__ import annotations
import math
import sys
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection

try:
    from scipy.spatial.distance import pdist, squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen

import uharfbuzz as hb

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


@dataclass
class StrokeSegment:
    """A single segment of a stroke with metadata"""
    start: Point
    end: Point
    length: float
    stroke_id: int
    segment_id: int


@dataclass
class Stroke:
    """A complete stroke with ordered segments"""
    segments: List[StrokeSegment]
    total_length: float
    closed: bool
    centroid: Point
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    stroke_id: int


class AdaptiveFlatteningPen(BasePen):
    """High-quality curve flattening with adaptive subdivision"""

    def __init__(self, glyphSet, tolerance: float = 0.1):
        super().__init__(glyphSet)
        self.tolerance = tolerance
        self.contours: List[Tuple[List[Point], bool]] = []
        self._current_contour: List[Point] = []
        self._start_point: Optional[Point] = None
        self._current_point: Optional[Point] = None

    def _moveTo(self, pt: Point) -> None:
        if self._current_contour:
            self.contours.append((self._current_contour[:], False))
        self._current_contour = [pt]
        self._start_point = pt
        self._current_point = pt

    def _lineTo(self, pt: Point) -> None:
        if self._current_point is None:
            self._moveTo(pt)
            return
        self._current_contour.append(pt)
        self._current_point = pt

    def _qCurveToOne(self, cp: Point, end: Point) -> None:
        """Quadratic Bézier curve with adaptive flattening"""
        if self._current_point is None:
            self._moveTo(cp)

        start = self._current_point
        points = self._flatten_quadratic(start, cp, end, self.tolerance)
        self._current_contour.extend(points[1:])  # Skip start point
        self._current_point = end

    def _curveToOne(self, cp1: Point, cp2: Point, end: Point) -> None:
        """Cubic Bézier curve with adaptive flattening"""
        if self._current_point is None:
            self._moveTo(cp1)

        start = self._current_point
        points = self._flatten_cubic(start, cp1, cp2, end, self.tolerance)
        self._current_contour.extend(points[1:])  # Skip start point
        self._current_point = end

    def _closePath(self) -> None:
        if self._current_contour and self._start_point:
            # Ensure closure
            if self._current_contour[-1] != self._start_point:
                self._current_contour.append(self._start_point)
            self.contours.append((self._current_contour[:], True))
        self._current_contour = []
        self._start_point = None
        self._current_point = None

    def _endPath(self) -> None:
        if self._current_contour:
            self.contours.append((self._current_contour[:], False))
        self._current_contour = []
        self._start_point = None
        self._current_point = None

    def addComponent(self, baseGlyphName: str, transformation) -> None:
        """Handle component glyphs with proper transformation"""
        try:
            base_glyph = self.glyphSet[baseGlyphName]
            transform_pen = TransformPen(self, transformation)
            base_glyph.draw(transform_pen)
        except KeyError:
            if DEBUG:
                print(f"Warning: Component glyph '{baseGlyphName}' not found")

    @staticmethod
    def _flatten_quadratic(p0: Point, p1: Point, p2: Point, tolerance: float) -> List[Point]:
        """Adaptively flatten a quadratic Bézier curve"""

        def subdivide(t0: float, t1: float, depth: int = 0) -> List[Point]:
            if depth > 12:  # Prevent infinite recursion
                return []

            # Evaluate curve at midpoint
            tm = (t0 + t1) * 0.5

            # De Casteljau's algorithm for quadratic curves
            def eval_quad(t: float) -> Point:
                u = 1 - t
                return (
                    u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
                )

            pt0 = eval_quad(t0)
            ptm = eval_quad(tm)
            pt1 = eval_quad(t1)

            # Check if linear approximation is good enough
            # Distance from midpoint to line segment
            line_mid = ((pt0[0] + pt1[0]) * 0.5, (pt0[1] + pt1[1]) * 0.5)
            dist = math.hypot(ptm[0] - line_mid[0], ptm[1] - line_mid[1])

            if dist <= tolerance:
                return [pt0, pt1]

            # Subdivide further
            left = subdivide(t0, tm, depth + 1)
            right = subdivide(tm, t1, depth + 1)

            return left[:-1] + right  # Avoid duplicate midpoint

        result = subdivide(0.0, 1.0)
        return result if result else [p0, p2]

    @staticmethod
    def _flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point, tolerance: float) -> List[Point]:
        """Adaptively flatten a cubic Bézier curve"""

        def subdivide(t0: float, t1: float, depth: int = 0) -> List[Point]:
            if depth > 12:
                return []

            tm = (t0 + t1) * 0.5

            # De Casteljau's algorithm for cubic curves
            def eval_cubic(t: float) -> Point:
                u = 1 - t
                return (
                    u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
                    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
                )

            pt0 = eval_cubic(t0)
            ptm = eval_cubic(tm)
            pt1 = eval_cubic(t1)

            # Flatness test using control polygon deviation
            line_mid = ((pt0[0] + pt1[0]) * 0.5, (pt0[1] + pt1[1]) * 0.5)
            dist = math.hypot(ptm[0] - line_mid[0], ptm[1] - line_mid[1])

            if dist <= tolerance:
                return [pt0, pt1]

            left = subdivide(t0, tm, depth + 1)
            right = subdivide(tm, t1, depth + 1)

            return left[:-1] + right

        result = subdivide(0.0, 1.0)
        return result if result else [p0, p3]


class StrokeOrderer:
    """Intelligently order strokes for natural handwriting appearance"""

    def __init__(self, strokes: List[Stroke]):
        self.strokes = strokes
        self.ordered_strokes: List[Stroke] = []

    def order_strokes(self) -> List[Stroke]:
        """Order strokes using multiple heuristics"""
        if not self.strokes:
            return []

        # Separate closed and open strokes
        closed_strokes = [s for s in self.strokes if s.closed]
        open_strokes = [s for s in self.strokes if not s.closed]

        ordered = []

        # Process closed strokes first (usually outer contours)
        if closed_strokes:
            ordered.extend(self._order_closed_strokes(closed_strokes))

        # Then process open strokes
        if open_strokes:
            ordered.extend(self._order_open_strokes(open_strokes))

        return ordered

    def _order_closed_strokes(self, strokes: List[Stroke]) -> List[Stroke]:
        """Order closed strokes: draw larger outer contours first, then smaller holes.
        This yields natural order (e.g., base shape before dot/inner counter)."""
        if not strokes:
            return []

        def stroke_area(stroke: Stroke) -> float:
            # Approximate area using bounding box
            min_x, min_y, max_x, max_y = stroke.bounds
            return (max_x - min_x) * (max_y - min_y)

        # Primary sort: by area (larger first)
        # Secondary: by X position (left-to-right)
        return sorted(strokes, key=lambda s: (-stroke_area(s), s.centroid[0]))

    def _order_open_strokes(self, strokes: List[Stroke]) -> List[Stroke]:
        """Order open strokes by position: left-to-right, then top-to-bottom."""
        if not strokes:
            return []

        # Sort primarily by X (left-to-right), then by Y (top-to-bottom)
        return sorted(strokes, key=lambda s: (s.centroid[0], -s.centroid[1]))


class HandwritingAnimator:
    """Main class for creating handwriting animations"""

    def __init__(self, font_path: str, text: str, font_size: float = 400):
        self.font_path = font_path
        self.text = text
        self.font_size = font_size

        # Load font and shape text
        self.tt_font = TTFont(font_path)
        self.hb_face, self.hb_font = self._load_harfbuzz_font()
        self.shaped_info = self._shape_text()

        # Extract font metrics
        self.units_per_em = self.hb_face.upem
        self.scale = font_size / self.units_per_em
        self._extract_font_metrics()

        # Generate strokes grouped per glyph (preserve HarfBuzz shaping order)
        self.grouped_strokes = self._extract_strokes_grouped()
        # Flatten for stats/debug
        self.strokes = [s for group in self.grouped_strokes for s in group]

        # Determine ordering mode: glyph-first preserves word order
        groups = self.grouped_strokes
        # Resolve writing direction
        resolved_dir = 'LTR'
        try:
            if isinstance(WRITING_DIRECTION, str) and WRITING_DIRECTION.upper() in ('LTR', 'RTL'):
                resolved_dir = WRITING_DIRECTION.upper()
            else:
                hb_dir = getattr(self, '_hb_direction', 'ltr')
                resolved_dir = 'RTL' if str(hb_dir).lower().startswith('rtl') else 'LTR'
        except Exception:
            resolved_dir = 'LTR'

        if resolved_dir == 'RTL':
            groups = list(reversed(groups))

        ordered: List[Stroke] = []
        if ORDER_MODE == 'glyph':
            for group in groups:
                if not group:
                    continue
                ordered.extend(StrokeOrderer(group).order_strokes())
        else:
            # Fallback to global ordering (not recommended for handwriting)
            ordered = StrokeOrderer(self.strokes).order_strokes()

        self.ordered_strokes = ordered

        if DEBUG:
            try:
                print(f"Resolved direction: {resolved_dir}; ORDER_MODE={ORDER_MODE}; groups={len(groups)}; total_strokes={len(self.strokes)}; ordered={len(self.ordered_strokes)}")
            except Exception:
                pass

        # Calculate total length for animation and cumulative per-stroke ends
        self.total_length = sum(stroke.total_length for stroke in self.ordered_strokes)
        cum = 0.0
        self.stroke_cum_ends: List[float] = []
        for s in self.ordered_strokes:
            cum += s.total_length
            self.stroke_cum_ends.append(cum)

    def _load_harfbuzz_font(self) -> Tuple[hb.Face, hb.Font]:
        """Load font for text shaping"""
        with open(self.font_path, 'rb') as f:
            font_data = f.read()
        face = hb.Face(font_data)
        font = hb.Font(face)
        return face, font

    def _shape_text(self) -> Tuple[List, List]:
        """Shape text using HarfBuzz"""
        buf = hb.Buffer()
        buf.add_str(self.text)
        buf.guess_segment_properties()

        # Enable comprehensive shaping features
        features = {
            "liga": True,  # Standard ligatures
            "clig": True,  # Contextual ligatures
            "kern": True,  # Kerning
            "calt": True,  # Contextual alternates
            "ccmp": True,  # Glyph composition/decomposition
        }

        hb.shape(self.hb_font, buf, features)

        # Persist useful shaping properties
        try:
            self._hb_direction = str(buf.direction)
            self._hb_script = str(buf.script)
            self._hb_language = str(buf.language)
        except Exception:
            self._hb_direction = 'ltr'
            self._hb_script = 'latn'
            self._hb_language = 'en'

        return buf.glyph_infos, buf.glyph_positions

    def _extract_font_metrics(self) -> None:
        """Extract comprehensive font metrics"""
        # Get metrics tables
        hhea = self.tt_font.get('hhea')
        os2 = self.tt_font.get('OS/2')

        if hhea:
            self.ascender = hhea.ascent * self.scale
            self.descender = hhea.descent * self.scale
            self.line_gap = hhea.lineGap * self.scale
        else:
            # Fallback values
            self.ascender = self.font_size * 0.8
            self.descender = -self.font_size * 0.2
            self.line_gap = 0

        # Calculate baseline position
        self.baseline_y = self.ascender if INVERT_Y else 0

        if DEBUG:
            print(f"Font metrics: ascender={self.ascender:.1f}, descender={self.descender:.1f}")

    def _extract_strokes(self) -> List[Stroke]:
        """Extract all strokes from shaped text (legacy flat list)."""
        infos, positions = self.shaped_info
        strokes = []

        x_cursor = 0.0

        for glyph_idx, (info, pos) in enumerate(zip(infos, positions)):
            glyph_id = info.codepoint

            # Calculate glyph position
            x_offset = pos.x_offset * self.scale
            y_offset = pos.y_offset * self.scale
            x_advance = pos.x_advance * self.scale

            glyph_x = x_cursor + x_offset
            glyph_y = self.baseline_y + y_offset

            # Extract glyph contours
            glyph_strokes = self._extract_glyph_strokes(
                glyph_id, glyph_x, glyph_y, len(strokes)
            )
            strokes.extend(glyph_strokes)

            x_cursor += x_advance

        return strokes

    def _extract_strokes_grouped(self) -> List[List[Stroke]]:
        """Extract strokes from shaped text, grouped by glyph in shaping order.
        Returns a list of groups; each group is a list[Stroke] for a single shaped glyph.
        """
        infos, positions = self.shaped_info
        grouped: List[List[Stroke]] = []
        x_cursor = 0.0
        stroke_id_offset = 0

        for glyph_idx, (info, pos) in enumerate(zip(infos, positions)):
            glyph_id = info.codepoint

            # Calculate glyph position
            x_offset = pos.x_offset * self.scale
            y_offset = pos.y_offset * self.scale
            x_advance = pos.x_advance * self.scale

            glyph_x = x_cursor + x_offset
            glyph_y = self.baseline_y + y_offset

            # Extract glyph contours
            glyph_strokes = self._extract_glyph_strokes(
                glyph_id, glyph_x, glyph_y, stroke_id_offset
            )
            grouped.append(glyph_strokes)
            stroke_id_offset += len(glyph_strokes)

            x_cursor += x_advance

        return grouped

    def _extract_glyph_strokes(self, glyph_id: int, x_offset: float, y_offset: float, stroke_id_offset: int) -> List[
        Stroke]:
        """Extract strokes from a single glyph"""
        try:
            glyph_name = self.tt_font.getGlyphName(glyph_id)
            glyph_set = self.tt_font.getGlyphSet()
            glyph = glyph_set[glyph_name]
        except Exception as e:
            if DEBUG:
                print(f"Warning: Could not load glyph {glyph_id}: {e}")
            return []

        # Flatten glyph outline
        tolerance = CURVE_TOLERANCE / self.scale  # Convert to font units
        pen = AdaptiveFlatteningPen(glyph_set, tolerance)
        glyph.draw(pen)

        strokes = []
        for stroke_idx, (points, closed) in enumerate(pen.contours):
            if len(points) < 2:
                continue

            # Transform points to final coordinates
            transformed_points = []
            for x, y in points:
                final_x = x * self.scale + x_offset
                final_y = (-y if INVERT_Y else y) * self.scale + y_offset
                transformed_points.append((final_x, final_y))

            # Create stroke segments
            segments = []
            total_length = 0.0

            for i in range(len(transformed_points) - 1):
                start = transformed_points[i]
                end = transformed_points[i + 1]
                length = math.hypot(end[0] - start[0], end[1] - start[1])

                segment = StrokeSegment(
                    start=start,
                    end=end,
                    length=length,
                    stroke_id=stroke_id_offset + stroke_idx,
                    segment_id=i
                )
                segments.append(segment)
                total_length += length

            if segments:
                # Calculate stroke properties
                all_x = [p[0] for p in transformed_points]
                all_y = [p[1] for p in transformed_points]

                centroid = (
                    sum(all_x) / len(all_x),
                    sum(all_y) / len(all_y)
                )

                bounds = (min(all_x), min(all_y), max(all_x), max(all_y))

                stroke = Stroke(
                    segments=segments,
                    total_length=total_length,
                    closed=closed,
                    centroid=centroid,
                    bounds=bounds,
                    stroke_id=stroke_id_offset + stroke_idx
                )
                strokes.append(stroke)

        return strokes

    def get_drawing_data_at_progress(self, progress: float) -> Tuple[np.ndarray, np.ndarray, Point]:
        """Get line segments and pen position for given progress (0.0 to 1.0)"""
        target_length = self.total_length * progress

        line_segments = []
        remaining_length = target_length
        pen_position = (0.0, 0.0)

        for stroke in self.ordered_strokes:
            if remaining_length <= 0:
                break

            if remaining_length >= stroke.total_length:
                # Include entire stroke
                for segment in stroke.segments:
                    line_segments.append([segment.start, segment.end])
                    pen_position = segment.end
                remaining_length -= stroke.total_length
            else:
                # Include partial stroke
                stroke_progress = 0.0
                for segment in stroke.segments:
                    if stroke_progress + segment.length <= remaining_length:
                        # Include entire segment
                        line_segments.append([segment.start, segment.end])
                        stroke_progress += segment.length
                        pen_position = segment.end
                    else:
                        # Include partial segment
                        remaining_seg_length = remaining_length - stroke_progress
                        if segment.length > 0:
                            t = remaining_seg_length / segment.length
                            partial_end = (
                                segment.start[0] + t * (segment.end[0] - segment.start[0]),
                                segment.start[1] + t * (segment.end[1] - segment.start[1])
                            )
                            line_segments.append([segment.start, partial_end])
                            pen_position = partial_end
                        break
                break

        if line_segments:
            segments_array = np.array(line_segments)
            return segments_array[:, 0], segments_array[:, 1], pen_position
        else:
            return np.array([]), np.array([]), pen_position

    def create_fill_patches(self) -> List[Tuple[int, PathPatch]]:
        """Create fill patches for closed contours.
        Returns list of (stroke_index, PathPatch) to enable per-stroke reveal.
        """
        patches: List[Tuple[int, PathPatch]] = []

        for idx, stroke in enumerate(self.ordered_strokes):
            if not stroke.closed or len(stroke.segments) < 3:
                continue

            # Create path vertices and codes
            vertices: List[Point] = []
            codes: List[int] = []

            # Start with first point
            vertices.append(stroke.segments[0].start)
            codes.append(MplPath.MOVETO)

            # Add all segment endpoints
            for segment in stroke.segments:
                vertices.append(segment.end)
                codes.append(MplPath.LINETO)

            # Close the path
            codes.append(MplPath.CLOSEPOLY)
            vertices.append(stroke.segments[0].start)

            # Create matplotlib path and patch
            path = MplPath(vertices, codes)
            patch = PathPatch(
                path,
                facecolor=FILL_COLOR,
                edgecolor='none',
                alpha=FILL_ALPHA,
                zorder=0
            )
            patches.append((idx, patch))

        return patches

    def animate(self, duration: float = 4.0, fps: int = 60) -> None:
        """Create and display the handwriting animation"""
        if not self.ordered_strokes:
            print("No strokes to animate!")
            return

        # Calculate figure size based on text bounds
        all_points = []
        for stroke in self.ordered_strokes:
            for segment in stroke.segments:
                all_points.extend([segment.start, segment.end])

        if all_points:
            all_x, all_y = zip(*all_points)
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            padding = self.font_size * 0.1
            width = (max_x - min_x) + 2 * padding
            height = (max_y - min_y) + 2 * padding
        else:
            min_x = max_x = min_y = max_y = 0
            width = height = self.font_size
            padding = self.font_size * 0.1

        # Create figure
        fig_width = max(width / FIGURE_DPI, MIN_FIGURE_WIDTH)
        fig_height = max(height / FIGURE_DPI, MIN_FIGURE_HEIGHT)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)

        # Set up plot
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(f'Handwriting: "{self.text}"', pad=20)

        # Initialize drawing elements
        line_collection = LineCollection(
            [],
            linewidths=STROKE_WIDTH,
            colors=STROKE_COLOR,
            linestyle='-',
            joinstyle=STROKE_JOIN,
            capstyle=STROKE_CAP,
            antialiased=True,
            zorder=1
        )
        ax.add_collection(line_collection)

        # Pen indicator
        pen_dot, = ax.plot([], [], 'o', color='red', markersize=8, zorder=2)

        # Fill patches (if enabled)
        fill_entries: List[Tuple[int, PathPatch]] = []
        closed_fill_by_index: Dict[int, PathPatch] = {}
        if SHOW_FILL:
            fill_entries = self.create_fill_patches()
            for idx, patch in fill_entries:
                ax.add_patch(patch)
                patch.set_visible(False)
            closed_fill_by_index = {idx: patch for idx, patch in fill_entries}

        # Animation state
        total_frames = int(duration * fps)
        current_frame = 0
        is_paused = False
        is_finished = False

        def update_frame():
            nonlocal current_frame, is_finished

            if is_paused or is_finished:
                return

            progress = min(current_frame / total_frames, 1.0)

            # Get drawing data
            starts, ends, pen_pos = self.get_drawing_data_at_progress(progress)

            # Update line collection
            if len(starts) > 0 and len(ends) > 0:
                segments = list(zip(starts, ends))
                line_collection.set_segments(segments)
            else:
                line_collection.set_segments([])

            # Update pen position
            pen_dot.set_data([pen_pos[0]], [pen_pos[1]])

            # Fill reveal logic
            if SHOW_FILL:
                if FILL_REVEAL == 'per_stroke':
                    target_len = self.total_length * progress
                    for idx, patch in closed_fill_by_index.items():
                        patch.set_visible(self.stroke_cum_ends[idx] <= target_len)
                elif FILL_REVEAL == 'global':
                    if progress >= FILL_THRESHOLD:
                        for patch in closed_fill_by_index.values():
                            patch.set_visible(True)
                # else 'none': do not reveal fills

            # Update frame
            current_frame += 1
            if current_frame > total_frames:
                is_finished = True

            fig.canvas.draw_idle()

        def restart_animation():
            nonlocal current_frame, is_paused, is_finished
            current_frame = 0
            is_paused = False
            is_finished = False

            # Hide fill patches
            for patch in closed_fill_by_index.values():
                patch.set_visible(False)

            line_collection.set_segments([])
            pen_dot.set_data([], [])
            fig.canvas.draw_idle()

        def toggle_pause():
            nonlocal is_paused
            is_paused = not is_paused

        # Set up timer
        timer = fig.canvas.new_timer(interval=int(1000 / fps))
        timer.add_callback(update_frame)

        # Keyboard controls
        def on_key(event):
            if event.key == ' ':
                toggle_pause()
            elif event.key in ['r', 'R']:
                restart_animation()
            elif event.key in ['q', 'Q', 'escape']:
                timer.stop()
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Control buttons
        button_height = 0.05
        button_width = 0.1
        button_y = 0.02

        replay_ax = fig.add_axes([0.78, button_y, button_width, button_height])
        replay_btn = Button(replay_ax, 'Replay')
        replay_btn.on_clicked(lambda x: restart_animation())

        exit_ax = fig.add_axes([0.89, button_y, button_width, button_height])
        exit_btn = Button(exit_ax, 'Exit')
        exit_btn.on_clicked(lambda x: (timer.stop(), plt.close(fig)))

        # Start animation
        restart_animation()
        timer.start()

        # Using explicit layout; tight_layout can conflict with custom axes/buttons
        plt.show()


# Configuration
TEXT = "Hello"
FONT_SIZE = 400
DURATION = 4.0
FPS = 60

# Stroke appearance
STROKE_WIDTH = 2.5
STROKE_COLOR = '#000000'
STROKE_JOIN = 'round'
STROKE_CAP = 'round'

# Sequencing and order controls
ORDER_MODE = 'glyph'       # 'glyph' (recommended) or 'global'
WRITING_DIRECTION = 'AUTO'  # 'AUTO', 'LTR', or 'RTL'

# Fill settings
SHOW_FILL = True
FILL_COLOR = '#FFDD44'  # Color for fill patches
FILL_ALPHA = 0.8
FILL_THRESHOLD = 0.95  # Used only for global fill reveal
FILL_REVEAL = 'per_stroke'  # 'per_stroke', 'global', or 'none'

# Technical parameters
CURVE_TOLERANCE = 0.5  # Curve flattening tolerance in pixels
INVERT_Y = False  # Font and Matplotlib both use y-up; avoid extra flipping

# Figure settings
FIGURE_DPI = 100
MIN_FIGURE_WIDTH = 8.0
MIN_FIGURE_HEIGHT = 4.0

# Debug settings
DEBUG = True

# Font management
FONT_DIR = Path(__file__).parent / 'fonts'


def find_fonts() -> List[Tuple[str, Path]]:
    """Find all available TrueType fonts"""
    if not FONT_DIR.exists():
        return []

    fonts = []
    for ext in ['*.ttf', '*.otf']:
        fonts.extend(FONT_DIR.glob(ext))

    return [(f.stem, f) for f in sorted(fonts)]


def select_font() -> Optional[Path]:
    """Interactive font selection"""
    fonts = find_fonts()

    if not fonts:
        print(f"No fonts found in {FONT_DIR}")
        return None

    if not sys.stdin.isatty():
        # Non-interactive mode
        return fonts[0][1]

    print("\nAvailable fonts:")
    for i, (name, path) in enumerate(fonts, 1):
        print(f"  {i:2d}. {name}")

    while True:
        try:
            choice = input(f"\nSelect font (1-{len(fonts)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(fonts):
                    return fonts[idx][1]
            print("Invalid selection, please try again.")
        except (EOFError, KeyboardInterrupt):
            return None


def main():
    """Main entry point"""
    print("Production Handwriting Animation")
    print("=" * 40)

    font_path = select_font()
    if not font_path:
        print("No font selected, exiting.")
        return

    print(f"\nUsing font: {font_path.name}")
    print(f"Animating text: '{TEXT}'")

    try:
        animator = HandwritingAnimator(str(font_path), TEXT, FONT_SIZE)

        if DEBUG:
            print(f"Generated {len(animator.strokes)} strokes")
            print(f"Total drawing length: {animator.total_length:.1f} pixels")

        animator.animate(DURATION, FPS)

    except Exception as e:
        print(f"Error creating animation: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()