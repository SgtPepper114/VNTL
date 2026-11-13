"""Selectable, wrapping Japanese text with real ruby above each word.

QLabel's HTML subset does not implement ruby. Paint the original text and
readings ourselves, while retaining QLabel's plain text/clipboard contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QFont, QFontMetricsF, QKeySequence, QPainter, QPalette
from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QSizePolicy

from furigana import RubySpan, annotate


@dataclass(frozen=True)
class _Cell:
    text: str
    reading: str
    start: int
    rect: QRectF
    ruby_height: float
    natural_width: float


class RubyLabel(QLabel):
    def __init__(self, text="", parent=None, *, furigana_enabled=True):
        super().__init__("", parent)
        self._furigana_enabled = furigana_enabled
        self._spans = ()
        self._layout_cache = {}
        self._anchor = 0
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setContentsMargins(0, 0, 0, 0)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setText(text)

    def setText(self, text):
        super().setText(text)
        self._refresh_spans()
        self.setAccessibleName(text)

    def clear(self):
        self.setText("")

    def setFuriganaEnabled(self, enabled):
        if self._furigana_enabled != bool(enabled):
            self._furigana_enabled = bool(enabled)
            self._refresh_spans()

    def _refresh_spans(self):
        self._spans = annotate(self.text()) if self._furigana_enabled else ()
        self._invalidate_layout()

    def _invalidate_layout(self):
        self._layout_cache.clear()
        self.updateGeometry()
        self.update()

    def _has_ruby(self):
        return any(span.reading for span in self._spans)

    def _fonts(self):
        base = self.font()
        ruby = QFont(base)
        if base.pointSizeF() > 0:
            ruby.setPointSizeF(max(5.0, base.pointSizeF() * 0.6))
        else:
            ruby.setPixelSize(max(6, round(base.pixelSize() * 0.55)))
        return base, ruby

    def _layout(self, width):
        width = max(1, int(width))
        if width in self._layout_cache:
            return self._layout_cache[width]
        base_font, ruby_font = self._fonts()
        fm, rm = QFontMetricsF(base_font), QFontMetricsF(ruby_font)
        units = []
        offset = 0
        for span in self._spans:
            parts = [span] if span.reading else [RubySpan(c) for c in span.text]
            for part in parts:
                display = "    " if part.text == "\t" else part.text
                w = max(fm.horizontalAdvance(display), rm.horizontalAdvance(part.reading))
                if part.reading:
                    w += 2  # Separate adjacent readings without changing the source.
                units.append((part, offset, max(0.1, w)))
                offset += len(part.text)
        rows, row, used = [], [], 0.0
        closing = "、。，．！？!?:：;；)]）］｝〉》」』】ぁぃぅぇぉっゃゅょァィゥェォッャュョー"
        opening = "([（［｛〈《「『【"
        for i, (span, start, natural) in enumerate(units):
            if span.text in ("\n", "\r"):
                if span.text == "\r" and i + 1 < len(units) and units[i + 1][0].text == "\n":
                    continue
                rows.append(row)
                row, used = [], 0.0
                continue
            cell_width = min(natural, width)
            required = cell_width
            # Keep closing punctuation with its preceding word where possible.
            for nxt, _, nxt_w in units[i + 1:]:
                if nxt.text not in closing:
                    break
                required += min(nxt_w, width)
            if span.text in opening and i + 1 < len(units):
                required += min(units[i + 1][2], width)
            if row and used + min(required, width) > width + 0.01:
                rows.append(row)
                row, used = [], 0.0
            row.append((span, start, natural, used, cell_width))
            used += cell_width
        rows.append(row)
        cells, y = [], 0.0
        for row in rows:
            ruby_height = rm.height() + 1 if any(s.reading for s, *_ in row) else 0
            line_height = math.ceil(ruby_height + fm.height() + 3)
            for span, start, natural, x, w in row:
                cells.append(_Cell(span.text, span.reading, start,
                                   QRectF(x, y, w, line_height), ruby_height, natural))
            y += line_height
        result = (tuple(cells), math.ceil(y))
        # Resize drags should not accumulate layouts indefinitely.
        if len(self._layout_cache) >= 8:
            self._layout_cache.clear()
        self._layout_cache[width] = result
        return result

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        if not self._has_ruby():
            return super().heightForWidth(width)
        margins = self.contentsMargins()
        return self._layout(width - margins.left() - margins.right())[1] + margins.top() + margins.bottom()

    def sizeHint(self):
        if not self._has_ruby():
            return super().sizeHint()
        width = max(100, min(640, super().sizeHint().width()))
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self):
        if not self._has_ruby():
            return super().minimumSizeHint()
        return QSize(1, self._layout(10000)[1])

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "_layout_cache") and event.type() in (
            QEvent.Type.FontChange, QEvent.Type.StyleChange, QEvent.Type.PaletteChange,
        ):
            self._invalidate_layout()

    def paintEvent(self, event):
        if not self._has_ruby():
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        area = self.contentsRect()
        painter.setClipRect(area)
        painter.translate(area.left(), area.top())
        base_font, ruby_font = self._fonts()
        fm, rm = QFontMetricsF(base_font), QFontMetricsF(ruby_font)
        palette = self.palette()
        text_color = palette.color(QPalette.ColorRole.WindowText)
        selection_start = self._selection_start()
        selection_end = selection_start + len(self.selectedText())
        for cell in self._layout(area.width())[0]:
            painter.save()
            painter.translate(cell.rect.left(), cell.rect.top())
            painter.scale(cell.rect.width() / cell.natural_width, 1)
            base_text = "    " if cell.text == "\t" else cell.text
            base_width = fm.horizontalAdvance(base_text)
            base_x = (cell.natural_width - base_width) / 2
            baseline = cell.ruby_height + fm.ascent()
            selected = selection_start < cell.start + len(cell.text) and selection_end > cell.start
            if selected:
                a = max(0, selection_start - cell.start)
                b = min(len(cell.text), selection_end - cell.start)
                x1 = base_x + fm.horizontalAdvance(cell.text[:a])
                x2 = base_x + fm.horizontalAdvance(cell.text[:b])
                painter.fillRect(QRectF(x1, cell.ruby_height, max(1, x2-x1), fm.height()),
                                 palette.brush(QPalette.ColorRole.Highlight))
            painter.setFont(base_font)
            painter.setPen(text_color)
            painter.drawText(QPointF(base_x, baseline), base_text)
            if cell.reading:
                painter.setFont(ruby_font)
                painter.drawText(QPointF((cell.natural_width-rm.horizontalAdvance(cell.reading))/2,
                                        rm.ascent()), cell.reading)
            painter.restore()
        painter.end()

    def _position_at(self, point):
        area = self.contentsRect()
        x, y = point.x() - area.left(), point.y() - area.top()
        cells = self._layout(area.width())[0]
        if not cells:
            return 0
        cell = min(cells, key=lambda c: (
            max(c.rect.top() - y, y - c.rect.bottom(), 0),
            max(c.rect.left() - x, x - c.rect.right(), 0)))
        fm = QFontMetricsF(self.font())
        scale = cell.rect.width() / cell.natural_width
        left = cell.rect.left() + (cell.natural_width - fm.horizontalAdvance(cell.text)) * scale / 2
        return cell.start + min(range(len(cell.text) + 1),
                                key=lambda i: abs(left + fm.horizontalAdvance(cell.text[:i]) * scale - x))

    def _selectable_ruby(self):
        return self._has_ruby() and bool(self.textInteractionFlags() &
                                        Qt.TextInteractionFlag.TextSelectableByMouse)

    def _select_span(self, start, length):
        """Our layout uses Unicode code points; Qt selections use UTF-16 units."""
        source = self.text()
        qt_start = len(source[:start].encode("utf-16-le")) // 2
        qt_length = len(source[start:start + length].encode("utf-16-le")) // 2
        self.setSelection(qt_start, qt_length)
        self.update()

    def _selection_start(self):
        start = self.selectionStart()
        if start < 0:
            return start
        return len(self.text().encode("utf-16-le")[:start * 2].decode("utf-16-le"))

    def mousePressEvent(self, event):
        if self._selectable_ruby() and event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self._anchor = self._position_at(event.position())
            self._select_span(self._anchor, 0)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selectable_ruby() and event.buttons() & Qt.MouseButton.LeftButton:
            end = self._position_at(event.position())
            self._select_span(min(self._anchor, end), abs(end - self._anchor))
        else:
            super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._selectable_ruby():
            pos = self._position_at(event.position())
            for cell in self._layout(self.contentsRect().width())[0]:
                if cell.start <= pos < cell.start + len(cell.text):
                    self._select_span(cell.start, len(cell.text))
                    return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if self._selectable_ruby() and event.matches(QKeySequence.StandardKey.SelectAll):
            self._select_span(0, len(self.text()))
        elif self._selectable_ruby() and event.matches(QKeySequence.StandardKey.Copy):
            QApplication.clipboard().setText(self.selectedText())
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        if not self._selectable_ruby():
            super().contextMenuEvent(event)
            return
        menu = QMenu(self)
        copy = menu.addAction("复制原文", lambda: QApplication.clipboard().setText(self.selectedText()))
        copy.setEnabled(bool(self.selectedText()))
        menu.addAction("全选", lambda: self._select_span(0, len(self.text())))
        menu.exec(event.globalPos())
