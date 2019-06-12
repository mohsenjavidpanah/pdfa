import os
import sys
import tempfile
import logging
from xml import sax

import cv2
import numpy as np
from pdf2image import convert_from_path


filter = False
logging.basicConfig(level=logging.INFO)
DEBUG = '--debug' in sys.argv
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)


class PDFaContentHandler(sax.handler.ContentHandler):
    def __init__(self, workspace_dir, page_images, *args, **kwargs):
        super(PDFaContentHandler, self).__init__(*args, **kwargs)
        self.__current_tag = None
        self.__current_attrs = {}
        self.__current_page = -1
        self.__current_page_erase_color = np.array([255, 255, 255], dtype=np.uint8)
        self.__current_page_image = None
        self.__hierarchy = []
        # self.workspace_dir = workspace_dir
        self.pages = []
        self.pages_images = page_images
        self.pages_cleaned_images = []

    def startElement(self, tag, attributes):
        attrs = dict(attributes)
        self.__current_tag = tag
        self.__current_attrs = attrs
        self.__hierarchy.append([tag, attrs])
        if tag == 'pdf2xml':
            pass
        elif tag == 'page':
            self.__current_page += 1
            self.pages.append([])
            self.__current_page_image = cv2.imread(
                os.path.join(self.pages_images[self.__current_page])
            )
            self.__current_page_image = cv2.resize(
                self.__current_page_image, (int(attrs['width']), int(attrs['height']))
            )
            self.__current_page_erase_color = self.__current_page_image[0, 0]
        elif tag == 'fontspec':
            pass
        elif tag == 'image':
            self.pages[-1].append([tag, attrs])
        elif tag in ['text', 'b']:
            self.pages[-1].append([tag, attrs, None])

    def characters(self, data):
        if self.__current_tag in ['b', 'text']:
            self.pages[-1][-1][2] = data

    def endElement(self, tag):
        if self.__current_tag in ['b', 'text', 'image'] and self.__current_attrs:
            x1 = int(self.__current_attrs['left'])
            y1 = int(self.__current_attrs['top'])
            x2 = int(self.__current_attrs['width']) + x1
            y2 = int(self.__current_attrs['height']) + y1
            self.__current_page_image = cv2.rectangle(
                self.__current_page_image,
                (x1, y1),
                (x2, y2),
                color=(255, 255, 255),
                thickness=cv2.FILLED
            )

        if tag == 'page':
            self.process_page()

        self.__hierarchy.pop()
        self.__current_tag = self.__hierarchy[-1][0] if len(self.__hierarchy) else None
        self.__current_attrs = self.__hierarchy[-1][1] if len(self.__hierarchy) else {}

    def detect_excepted_vertical_lines_in_table(self, table, gray_image):
        """
        Parameters
        ----------
        tabel : list
            A nested list of lines that categorized by Line Arrow direction

        gray_image: OpenCV.Image
            The gray image instance
        """
        max_vertical_x = 0
        for vrow_index in range(1, len(table), 2):
            max_vertical_x = max(
                sorted(
                    table[vrow_index],
                    key=lambda mr: mr[0],
                    reverse=True
                )[0][0],
                max_vertical_x
            )
        for vrow_index in range(1, len(table), 2):
            line_detected = False
            vrow = sorted(table[vrow_index], key=lambda r: r[0])
            # vrow = table[vrow_index]
            for vline_index in range(len(vrow)):
                vline = vrow[vline_index]
                next_vline = None
                if vline_index < len(vrow) - 1:
                    next_vline = vrow[vline_index + 1]
                else:
                    next_vline = [max_vertical_x]

                img_slice = gray_image[
                    vline[1]:vline[3],
                    vline[2] + 5:next_vline[0] - 5
                ]
                oldx = 0
                thickness = 0
                points = np.where(img_slice <= 220)
                for x in np.unique(points[1]):
                    if (img_slice[:, x] <= 220).sum() >= abs(vline[1] - vline[3]) / 1.4:
                        line_detected = True
                        thickness += 1
                    elif line_detected:
                        cv2.imwrite(
                            f'/home/mohsen/aaa/{vline[2]} {vline[1]} -'
                            f' {next_vline[0]} {vline[3]}.png', img_slice
                        )
                        if [vline[0] + oldx + 5, vline[1], vline[2] + oldx + 5, vline[3], 'V', thickness] \
                           not in table[vrow_index]:
                            x = vline[0] + oldx + 5
                            initiated_x = x
                            for same_vrow in table:
                                for x_vline in same_vrow:
                                    if abs(x_vline[0] - x) < 5:
                                        x = x_vline[0]
                                        break
                                if x != initiated_x:
                                    break
                            vrow.append(
                                [x, vline[1], x, vline[3], 'V', thickness]
                            )
                            table[vrow_index] = sorted(vrow, key=lambda r: r[0])
                        line_detected = False
                    oldx = x

        return table

    def split_multiple_cells_lines(self, table):
        """
        Split long lines that members of two or greater cells

        Parameters
        ----------
        tabel : list
            A nested list of lines that categorized by Line Arrow direction
        """
        all_h_cells = []
        all_v_cells = []
        for row in table:
            if row[0][4] == 'V':
                all_v_cells.extend(row[:])
            else:
                all_h_cells.extend(row[:])

        logging.debug(f'Q@ 1 ---> {table}')
        new_table = table[:]
        i = 0
        for row in table:
            j = 0
            splits = set()
            for line in row:
                if line[4] == 'H':
                    for vline in all_v_cells:
                        # if ((abs(vline[1] - line[1]) < 5) or (abs(vline[3] - line[1]) < 5)) and \
                        #    line[0] < vline[0] < line[2]:
                        if line[0] < vline[0] < line[2]:
                            splits.add(vline[0])
                    y = line[1]
                    splits = sorted(splits)
                    logging.debug(f'HHHHHHHHHHHHHHHHHHHHHHHHH {splits}')
                    if splits:
                        old_line = new_table[i][j]
                        del new_table[i][j]
                        thickness = 1
                        start_x = line[0]
                        for x in splits:
                            if start_x == x - 1:
                                thickness += 1
                                logging.debug('H |||> 1')
                            else:
                                new_table[i].insert(j, [start_x, y, x, y, 'H', thickness or 1])
                                logging.debug('H |||> 2')
                                thickness = 0
                            start_x = x
                        if len(splits) == 0:
                            new_table[i].insert(j, old_line)
                        else:
                            new_table[i].insert(j, [start_x, y, line[2], y, 'H', thickness or 1])
                    splits = set()
                elif line[4] == 'V':
                    # splits.add(line[1])
                    for hline in all_h_cells:
                        # if ((abs(hline[0] - line[0]) < 5) or (abs(hline[2] - line[0]) < 5)) and \
                        #    line[1] < hline[1] < line[3]:
                        if line[1] < hline[1] < line[3] and abs(line[3] - hline[1]) > 5 and \
                           abs(line[1] - hline[1]) > 5:
                            splits.add(hline[1])
                    splits.add(line[3])
                    x = line[0]
                    splits = sorted(splits)
                    logging.debug(f'>>>>>>>>> {i} {len(new_table)} {new_table} {splits}')
                    old_line = new_table[i][j]
                    start_y = line[1]
                    if splits and splits[0] != old_line[3]:
                        logging.debug(f'0 >>> {splits} {[line[1], line[3]]}')
                        logging.debug(f'1 >>> {new_table[i]}')
                        del new_table[i][j]
                        for y in splits:
                            # Split Vertical Line in Others Row
                            vinserted = False
                            min_dist = None
                            min_vrow = None
                            vindex = 0
                            for vrow in new_table:
                                if vinserted:
                                    break

                                for vline in vrow:
                                    if vline[4] != 'V':
                                        continue

                                    dist = (abs(vline[1] - start_y))
                                    logging.debug(f'2 >>> {start_y} {vline}')
                                    # import pdb; pdb.set_trace()
                                    if dist <= 5:
                                        vrow.append([x, start_y, x, y, 'V', 1])
                                        logging.debug(f'2.1 new_table >>> {new_table}')
                                        start_y = y
                                        vinserted = True
                                        break
                                    elif dist < (min_dist or dist + 1):
                                        logging.debug(f'2.2 new_table >>> {new_table}')
                                        min_dist = dist
                                        min_vrow = vrow
                                    # else:
                                    #     new_table[i].append([x, start_y, x, y, 'V', 1])
                                    #     logging.debug('2.3 len(vrow) >>>', len(vrow))

                                vindex += 1

                            if not vinserted:
                                # Split Vertical Line in Others Row
                                min_hrows = {}
                                hindex = 0
                                hrow_index = 0

                                for hrow in new_table:
                                    for hline in hrow:
                                        if hline[4] != 'H':
                                            continue

                                        dist = (abs(hline[1] - start_y))
                                        logging.debug(f'3 >>> {start_y} {hline} {dist}')

                                        min_hrows[dist] = hrow_index
                                        break

                                    hrow_index += 1

                                if min_hrows:
                                    min_row = min(*min_hrows.keys())
                                    hrow_index = min_hrows[min_row]
                                    logging.debug(f'4 >>> {start_y} {hline} {hrow_index}')
                                    insert_after = True
                                    if new_table[hrow_index][0][1] < start_y or (hrow_index + 1) == len(new_table):
                                        if (hrow_index + 1) == len(new_table):
                                            hrow_index -= 2
                                        # if hrow_index == 0 and new_table[hrow_index][0][4] == 'H':
                                        #     hrow_index += 1
                                        #     new_table.insert(0, [])

                                        new_table[hrow_index + 1].append([x, start_y, x, y, 'V', 1])
                                        logging.debug(f'4.1 new_table >>> {new_table}')
                                        start_y = y
                                        vinserted = True
                                        if len(new_table[i]) == 0:
                                            del new_table[i]
                                        insert_after = False

                                    if insert_after is True:
                                        if len(new_table) < (hrow_index + 2):
                                            new_table.append([])
                                        new_table[hrow_index + 1].append([x, start_y, x, y, 'V', 1])
                                        logging.debug(f'4.2 new_table >>> {new_table}')
                                        start_y = y
                                        vinserted = True
                                        if len(new_table[i]) == 0:
                                            del new_table[i]

                                logging.debug(f'5 >>> {min_vrow}')
                                if not vinserted and \
                                   [x, min_vrow[0][1], x, min_vrow[0][3], 'V', 1] not in min_vrow:
                                    min_vrow.append([x, min_vrow[0][1], x, min_vrow[0][3], 'V', 1])

                        if len(splits) == 0:
                            new_table[i].insert(j, old_line)
                    splits = set()
                j += 1
            i += 1

        i = 0
        while i < len(new_table):
            row = new_table[i]
            logging.debug(f'6 >>> {row}')
            # if len(row) == 0:
            #     del new_table[i]
            #     continue
            if row[0][4] == 'H':
                new_table[i] = sorted(row, key=lambda l: [l[0], l[1]])
            elif row[0][4] == 'V':
                new_table[i] = sorted(row, key=lambda l: [l[1], l[0]])
            i += 1

        logging.debug(f'Q@ 2 ---> {table}')

        return new_table

    # def detect_precise_lines_position_and_style(self, table, gray_image):
    #     for row in table:
    #         for line in row:
    #             if line[4] == 'H':
    #                 img_slice = gray_image[
    #                     line[1] - 2: line[1] + 2,
    #                     line[0] + 2: line[2] - 2
    #                 ]
    #                 row_indices = np.unique(np.where(img_slice != self.__current_page_erase_color)[0])
    #                 valid_indices = set()
    #                 for index in row_indices:
    #                     count = len(img_slice[index][img_slice[index] != self.__current_page_erase_color])
    #                     if count >= img_slice.shape[1] / 1.4:
    #                         valid_indices.add(index)
    #                 if valid_indices:
    #                     # TODO: Must detect line style in future same as dashes, double lines, etc...
    #                     line[1] = line[3] = min(valid_indices) - 2 + line[1]
    #                 # import pdb; pdb.set_trace()
    #                 # pass
    #             else:  # line[4] == 'V'
    #                 img_slice = gray_image[
    #                     line[1] + 2: line[1] - 2,
    #                     line[0] - 2:line[2] + 2]
    #                 col_indices = np.unique(np.where(img_slice != self.__current_page_erase_color)[1])
    #                 valid_indices = set()
    #                 for index in col_indices:
    #                     count = len(
    #                         img_slice[:, index][img_slice[:, index] != self.__current_page_erase_color]
    #                     )
    #                     if count >= img_slice.shape[0] / 1.4:
    #                         valid_indices.add(index)
    #                 if valid_indices:
    #                     # TODO: Must detect line style in future same as dashes, double lines, etc...
    #                     line[0] = line[2] = min(valid_indices) - 2 + line[0]
    #     return table

    def process_page(self):
        cv2.imwrite(
            os.path.splitext(self.pages_images[self.__current_page])[0] + '-out.jpg',
            self.__current_page_image
        )
        # detect table lines
        gray = cv2.cvtColor(self.__current_page_image, cv2.COLOR_BGR2GRAY)
        self.__current_page_erase_color = gray[0, 0]
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(gray)[0]
        final_lines = [
            [
                int(round(p[0])), int(round(p[1])),
                int(round(p[2])), int(round(p[3]))
            ]
            for p in (line[0] for line in lines)
        ]
        final_lines = sorted(final_lines, key=lambda l: [l[1], l[0]])
        #
        index1 = 0
        oldy = -1
        xs = set()
        ys = set()
        tables_y = []
        table = []
        table_info = {}
        tables = []
        tables_info = []
        registered_lines = []
        arrow = ''
        old_arrow = ''
        line_row = []
        deleted = False
        while index1 < len(final_lines):
            line1 = final_lines[index1]
            x1, y1, x2, y2 = line1
            # First point of line must be lowest of second point
            if (y1 - y2 > 5) or (abs(y1 - y2) < 5 and x1 - x2 > 5):
                ty = y1
                tx = x1
                y1 = y2
                x1 = x2
                y2 = ty
                x2 = tx

                line1[0] = x1
                line1[1] = y1
                line1[2] = x2
                line1[3] = y2

            # Find and separate tables in vertical arrow
            new_table = (len(tables_y) == 0 or (y1 - tables_y[-1]) >= 5) and y1 not in tables_y
            for item in ys:
                if not new_table:
                    break
                if (len(ys) and (y1 - item) < 5):
                    new_table = False

            if new_table:
                if line_row:
                    table.append(line_row)
                line_row = []
                table = []
                table_info = {
                    'min-x': min(x1, x2),
                    'min-y': min(y1, y2),
                    'max-x': max(x1, x2),
                    'max-y': max(y1, y2)
                }
                tables.append(table)
                tables_info.append(table_info)
                tables_y.append(y1)
                gray = cv2.putText(
                    gray, str(len(tables_y)), (5, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA
                )

            table_info.update({
                'min-x': min(min(x1, x2), table_info['min-x']),
                'min-y': min(min(y1, y2), table_info['min-y']),
                'max-x': max(max(x1, x2), table_info['max-x']),
                'max-y': max(max(y1, y2), table_info['max-y'])
            })

            xs.add(x1)
            xs.add(x2)
            ys.add(y1)
            ys.add(y2)
            if oldy != y1:
                oldy = y1

            if x1 != x2 and abs(x1 - x2) <= 5:
                x1 = max(x1, x2)
                x2 = x1
                final_lines[index1][0] = x1
                final_lines[index1][2] = x2
            elif y1 != y2 and abs(y1 - y2) <= 5:
                y1 = min(y1, y2)
                y2 = y1
                final_lines[index1][1] = y1
                final_lines[index1][3] = y2

            _arrow = 'V' if x1 == x2 else 'H'
            arrow = arrow or _arrow
            if not deleted:
                old_arrow = arrow
            arrow = _arrow

            index2 = 0
            deleted = False
            diff = 0
            while index2 < len(final_lines):
                if index1 == index2:
                    index2 += 1
                    continue

                line2 = final_lines[index2]
                x3, y3, x4, y4 = line2

                # if abs(y1 - y3) > 10:
                #     break

                if y1 == y2 and y3 == y4:  # Horizontal Lines
                    diff = abs(y1 - y3)
                elif x1 == x2 and x3 == x4:  # Vertical Lines
                    diff = abs(x1 - x3)

                if diff and diff < 5:
                    del final_lines[index2]
                    deleted = True
                    break
                else:
                    index2 += 1

            if not deleted:
                # OpenCV Generates Two Line per Real One Line
                # We need To Remove One Of Them
                # if line1 not in registered_lines:
                registered_lines.append(line1)
                # import pdb; pdb.set_trace()
                if old_arrow != arrow:
                    if line_row:
                        table.append(line_row)
                    line_row = []
                line_row.append(line1 + [arrow])

                index1 += 1

        if line_row:
            table.append(line_row)

        # Enhancing Lines connection spots
        ti = 0
        index = 0
        tables_data = []
        old_arrow = ''

        def process_row_lines(lines, data):
            for line in lines:
                if abs(line[0] - data['min-x']) < 5:
                    line[0] = data['min-x']
                if abs(line[1] - data['min-y']) < 5:
                    line[1] = data['min-y']
                if abs(line[2] - data['max-x']) < 5:
                    line[2] = data['max-x']
                if abs(line[3] - data['max-y']) < 5:
                    line[3] = data['max-y']

        for table in tables:
            logging.debug(f'\n\n\n\nTABLE --- :> {table}')
            # Reordering Table Rows First
            for i in range(1, len(table), 2):
                r1 = table[i]
                if len(r1) == 0:
                    del table[i]
                    continue
                for j in range(i + 2, len(table), 2):
                    r2 = table[j]
                    if r1 and r2:
                        for l1 in range(len(r1)):
                            for l2 in range(len(r2)):
                                if abs(r1[l1][1] - r2[l2][1]) < 10:
                                    r1.append(r2[l2])
                                    del r2[l2]

            k = 1
            while k < len(table) - 1:
                if len(table[k]) == 0:
                    if table[k - 1][0][4] == table[k + 1][0][4]:
                        table[k - 1] += table[k + 1]
                        del table[k]
                        del table[k]
                        continue
                k += 1


            logging.debug(f'\n\nTABLE :> {table}\n\n\n\n\n')
            i = 0
            while i < len(table) - 1:
                if table[i][0][4] == table[i + 1][0][4]:
                    table[i].extend(table[i + 1])
                    del table[i + 1]
                    continue
                i += 1

            # Split multiple cells Lines
            table = self.split_multiple_cells_lines(table)
            # Detect excepted vertical lines
            table = self.detect_excepted_vertical_lines_in_table(table, gray)
            # Detect precise lines position
            # table = self.detect_precise_lines_position_and_style(table, gray)

            data = {'cells': []}
            data['min-x'] = tables_info[ti]['min-x']
            data['min-y'] = tables_info[ti]['min-y']
            data['max-x'] = tables_info[ti]['max-x']
            data['max-y'] = tables_info[ti]['max-y']
            ti += 1
            process_row_lines(table[0], data)
            process_row_lines(table[-1], data)
            lines = table[0][:]
            is_real_table = True
            next_row = []
            vlines = []
            for i in range(1, len(table) - 1, 2):
                process_row_lines(table[i], data)
                prev_row = table[i - 1]
                row = table[i]
                next_row = table[i + 1]
                process_row_lines(row, data)
                # old_line = row[0]
                for line_index in range(len(row)):
                    line = row[line_index]
                    x1, y1, x2, y2 = line[:4]
                    if line[4] != 'V':
                        raise Exception('Vertical!!?')
                    if abs(y1 - data['min-y']) < 10:
                        line[1] = data['min-y']
                    # elif abs(y1 - old_line[3]) < 10:
                    #     line[1] = old_line[3]

                    for l in prev_row:
                        if abs(y1 - l[1]) < 15:
                            line[1] = l[1] + 50
                        if abs(y1 - l[3]) < 15:
                            line[1] = l[3]
                        if abs(y2 - l[1]) < 15:
                            line[3] = l[1] + 50
                        if abs(y2 - l[3]) < 15:
                            line[3] = l[3]
                        if abs(x1 - l[0]) < 15:
                            l[0] = x1
                        elif abs(x1 - l[2]) < 15:
                            l[2] = x1

                    for l in next_row:
                        if abs(y2 - l[3]) < 15:
                            line[3] = l[3]
                        elif abs(y2 - l[1]) < 15:
                            line[3] = l[1]
                        if abs(y1 - l[1]) < 15:
                            line[1] = l[1] + 50
                        if abs(y1 - l[3]) < 15:
                            line[1] = l[3]
                        if abs(x1 - l[0]) < 15:
                            l[0] = x1
                        if abs(x1 - l[2]) < 15:
                            l[2] = x1

                    vlines.append(line)

                    # else:
                    #     if abs(x1 - data['min-x']) < 10:
                    #         line[0] = data['min-x']
                    #     elif abs(x1 - old_line[0]) < 10:
                    #         line[0] = old_line[0]
                    # old_line = line

            for last_line in next_row:
                if last_line and last_line[1] != last_line[3]:
                    last_line[1] = max(last_line[1], last_line[3])
                    last_line[3] = last_line[1]

            for j in range(0, len(table), 2):
                hrow = table[j]
                for hline in hrow:
                    for vline in vlines:
                        if abs(hline[0] - vline[0]) < 10:
                            hline[0] = vline[0]
                        if abs(hline[2] - vline[2]) < 10:
                            hline[2] = vline[2]

            # Detect excepted Horizontal Lines
            for hrow_index in range(0, len(table), 2):
                hrow = table[hrow_index]
                for hline_index in range(len(hrow)):
                    hline = hrow[hline_index]

                    next_hline = None
                    if hline_index < len(hrow) - 1:
                        next_hline = hrow[hline_index + 1]
                    else:
                        next_hline = [data['max-x']]

                    if [hline[2], hline[3], next_hline[0], hline[3]] in hrow:
                        continue

                    if next_hline[0] - hline[2] > 2:
                        img_slice = gray[
                            hline[1] - 4:hline[1] + 4,
                            hline[2] + 2:next_hline[0] - 2
                        ].copy()
                        img_slice[img_slice == self.__current_page_erase_color] = 0
                        img_slice[img_slice != 0] = 255

                        counts = 0
                        for y in np.unique(np.nonzero(img_slice)[0]):
                            counts = np.count_nonzero(img_slice[y])
                            # Maybe next time in others PDF pages complexest condition raisin
                            # But in current time i dont consider these
                            if 5 < counts >= img_slice.shape[1] / 1.4:
                                hrow.insert(
                                    hline_index,
                                    [
                                        hline[2], hline[3],
                                        next_hline[0], hline[3], 'H'
                                    ]
                                )
                                break

            if not is_real_table:
                del tables[index]
            else:
                tables_data.append(data)
                tables_data[index]['cells'] = self.detect_cells(table)
                tables[index] = table
            index += 1

        n = 0
        for table in tables:
            for row in table:
                for line in row:
                    n += 1
                    x0, y0, x1, y1 = line[:4]
                    gray = cv2.line(gray, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imwrite(
            os.path.splitext(self.pages_images[self.__current_page])[0] + '-out-gray.jpg', gray
        )

    nimage = 0

    def detect_cells(self, table):
        hlines = np.array(None, None)
        vlines = np.array(None, None)
        points = []
        # points = np.array(None, None)
        img = self.__current_page_image.copy()
        for row in table:
            for line in row:
                point1 = np.array([line[0], line[1]])
                point2 = np.array([line[2], line[3]])
                if not points or \
                   (points and np.array(
                       [np.abs(np.linalg.norm(np.array(p) - point1)) for p in points]
                   ).min() > 5):
                    points.append(point1)
                    img[point1[1], point1[0]] = [255, 0, 0]
                    # logging.debug([point1[0], point1[1]], img[point1[1], point1[0]])
                if points and np.array(
                        [np.abs(np.linalg.norm(np.array(p) - point2)) for p in points]).min() > 5:
                    points.append(point2)
                    img[point2[1], point2[0]] = [255, 0, 0]
                    # logging.debug([point2[0], point2[1]], img[point2[1], point2[0]])

                if line[4] == 'H':
                    hlines.put(0, np.array(line))
                else:  # line[4] == 'V'
                    vlines.put(0, np.array(line))

        cv2.imwrite(f'/home/mohsen/aax-{self.nimage}.png', img)
        self.nimage += 1
        return []


class PDFaParser(object):
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf = open(pdf_path)
        self.pages = []
        self.temp = tempfile.mkdtemp(prefix="pidifa_")
        logging.info(self.temp + '/pages')

    def convert_pdf_to_docx(self):
        words, boxes = self.fold_pdf_style()

    def fold_pdf_style(self):
        # Extract XML
        tpath = os.path.join(self.temp, "pdf.xml")
        # logging.debug("----->", f"pdftohtml -c -xml {self.pdf_path} {tpath}")
        os.system(f"pdftohtml -c -xml {self.pdf_path} {tpath}")
        # Extract PDF Pages Images
        pages_dir = os.path.join(self.temp, 'pages')
        os.mkdir(pages_dir)
        self.pages = [
            p.filename for p in convert_from_path(self.pdf_path, output_folder=pages_dir, fmt="png")
        ]
        # Make Output
        with open(tpath) as xml_file:
            sax.parse(xml_file, PDFaContentHandler(self.temp, self.pages))
        return 0, 0


pdf_paths = sys.argv[1:]

for option in ['--debug']:
    if option in pdf_paths:
        del pdf_paths[pdf_paths.index(option)]

if not pdf_paths:
    pdf_paths = [os.path.join(os.getcwd(), '2.pdf')]

for pdf_path in pdf_paths:
    pdfp = PDFaParser(pdf_path)
    pdfp.convert_pdf_to_docx()


# file_path = '2.jpg'
# img = cv2.imread(file_path)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 90, 150, apertureSize=3)
# kernel = np.ones((3, 3), np.uint8)
# edges = cv2.dilate(edges, kernel, iterations=1)
# kernel = np.ones((5, 5), np.uint8)
# edges = cv2.erode(edges, kernel, iterations=1)
# cv2.imwrite('canny.jpg', edges)

# lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

# if not lines.any():
#     logging.debug('No lines were found')
#     exit()

# if filter:
#     rho_threshold = 15
#     theta_threshold = 0.1

#     # how many lines are similar to a given one
#     similar_lines = {i : [] for i in range(len(lines))}
#     for i in range(len(lines)):
#         for j in range(len(lines)):
#             if i == j:
#                 continue

#             rho_i, theta_i = lines[i][0]
#             rho_j, theta_j = lines[j][0]
#             if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
#                 similar_lines[i].append(j)

#     # ordering the INDECES of the lines by how many are similar to them
#     indices = [i for i in range(len(lines))]
#     indices.sort(key=lambda x : len(similar_lines[x]))

#     # line flags is the base for the filtering
#     line_flags = len(lines)*[True]
#     for i in range(len(lines) - 1):
#         if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
#             continue

#         for j in range(i + 1,  len(lines)): # we are only considering those elements that had less similar line
#             if not line_flags[indices[j]]: # and only if we have not disregarded them already
#                 continue

#             rho_i, theta_i = lines[indices[i]][0]
#             rho_j, theta_j = lines[indices[j]][0]
#             if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
#                 line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

# logging.debug('number of Hough lines:', len(lines))

# filtered_lines = []

# if filter:
#     for i in range(len(lines)): # filtering
#         if line_flags[i]:
#             filtered_lines.append(lines[i])

#     logging.debug('Number of filtered lines:',  len(filtered_lines))
# else:
#     filtered_lines = lines

# for line in filtered_lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imwrite('hough.jpg', img)
