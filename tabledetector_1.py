import os
import sys
import tempfile
from xml import sax

import cv2
import numpy as np
from pdf2image import convert_from_path


filter = False


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
            x2 = int(self.__current_attrs['width'])
            y2 = int(self.__current_attrs['height'])
            self.__current_page_image = cv2.rectangle(
                self.__current_page_image,
                (x1, y1, x2, y2),
                color=(255, 255, 255),
                thickness=cv2.FILLED
            )

        if tag == 'page':
            self.process_page()

        self.__hierarchy.pop()
        self.__current_tag = self.__hierarchy[-1][0] if len(self.__hierarchy) else None
        self.__current_attrs = self.__hierarchy[-1][1] if len(self.__hierarchy) else {}

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
            i = 0
            while i < len(table) - 1:
                if table[i][0][4] == table[i + 1][0][4]:
                    table[i].extend(table[i + 1])
                    del table[i + 1]
                    continue
                i += 1

            # Split Unique multiple Cells Vertical Lines
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
            1 + 1
            # print('----->>', ti, table)
            # for vrow_index in range(1, len(table), 2):
            #     vrow = sorted(table[vrow_index], key=lambda r: r[0])
            #     for vline_index in range(len(vrow)):
            #         vline = vrow[vline_index]
            #         next_vline = None
            #         if vline_index < len(vrow) - 1:
            #             next_vline = vrow[vline_index + 1]
            #         else:
            #             next_vline = [max_vertical_x]

            #         img_slice = gray[
            #             vline[1] + 2:vline[3] - 2,
            #             vline[2] + 2:next_vline[0] - 2
            #         ].copy()
            #         img_slice[img_slice == self.__current_page_erase_color] = 0
            #         img_slice[img_slice != 0] = 255
            #         counter = 0
            #         removed_index = []
            #         current_index = 0
            #         min_average = 100
            #         while current_index < len(img_slice):
            #             if np.average(img_slice[current_index]) <= min_average:
            #                 img_slice = np.delete(img_slice, current_index)
            #                 removed_index.append(counter)
            #             else:
            #                 current_index += 1
            #             counter += 1
            #         # Maybe next time in others PDF pages complexest condition raisin
            #         # But in current time i dont consider these
            #         if len(img_slice):
            #             vrow.insert(
            #                 vline_index,
            #                 [
            #                     vline[2], vline[3],
            #                     next_vline[0], vline[1]
            #                 ]
            #             )
            #             print('=======>>', [
            #                 vline[2], vline[3],
            #                 next_vline[0], vline[1]
            #             ])
            #     if (vrow[-1][0] - max_vertical_x) < 5:
            #         vline = vrow[-1]
            #         next_vline = None
            #         try:
            #             next_vline = vrow[-2]
            #         except IndexError:
            #             max_dist = 9999999999999999999999999
            #             for table_v_line in table:
            #                 if table[4] == 'V' and abs(table_v_line[0] - vline[0]) < max_dist and \
            #                    abs(table_v_line[1] - vline[1]) < 5:
            #                     next_vline = table_v_line
            #                     max_dist = abs(table_v_line[0] - vline[0])
            #             # if next_vline is None:
            #             #     continue
            #         img_slice = gray[
            #             vline[1] + 2:vline[3] - 2,
            #             next_vline[0] + 2:vline[2] - 2
            #         ].copy()
            #         img_slice[img_slice == self.__current_page_erase_color] = 0
            #         img_slice[img_slice != 0] = 255
            #         min_average = 255 if img_slice.shape[1] == 0 else 100 / img_slice.shape[1]
            #         counter = 0
            #         removed_index = []
            #         current_index = 0
            #         while current_index < len(img_slice):
            #             if np.average(img_slice[current_index]) <= min_average:
            #                 img_slice = np.delete(img_slice, current_index)
            #                 removed_index.append(counter)
            #             else:
            #                 current_index += 1
            #             counter += 1
            #         # Maybe next time in others PDF pages complexest condition raisin
            #         # But in current time i dont consider these
            #         if len(img_slice):
            #             vrow.insert(
            #                 vline_index,
            #                 [
            #                     vline[2], vline[3],
            #                     next_vline[0], next_vline[1]
            #                 ]
            #             )

            # print('+++++>>', ti, table)
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
            for i in range(1, len(table), 2):
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
                        # if ti >= 8 and i == 3:
                        #     print(ti, i, line_index, len(row))
                        #     import pdb; pdb.set_trace()
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

            # # Detect excepted Vertical Lines
            # for vrow_index in range(1, len(table), 2):
            #     vrow = sorted(table[vrow_index], key=lambda r: r[0])
            #     for vline_index in range(len(vrow)):
            #         vline = vrow[vline_index]
            #         next_vline = None
            #         if vline_index < len(vrow) - 1:
            #             next_vline = vrow[vline_index + 1]
            #         else:
            #             next_vline = [data['max-x']]

            #         img_slice = gray[
            #             vline[1] + 4:vline[3] - 4,
            #             vline[2] + 2:next_vline[0] - 2
            #         ].copy()
            #         img_slice[img_slice == self.__current_page_erase_color] = 0
            #         img_slice[img_slice != 0] = 255
            #         # Maybe next time in others PDF pages complexest condition raisin
            #         # But in current time i dont consider these
            #         aaq = np.unique(np.nonzero(img_slice)[1])
            #         if 5 < len(np.unique(np.nonzero(img_slice)[1])) >= \
            #           (next_vline[0] - vline[2] - 2) / 2:
            #             vrow.insert(
            #                 vline_index,
            #                 [
            #                     vline[2], vline[3],
            #                     next_vline[0], next_vline[1]
            #                 ]
            #             )

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

                    if next_hline[0] - hline[2] > 2:
                        img_slice = gray[
                            hline[1] - 4:hline[1] + 4,
                            hline[2] + 2:next_hline[0] - 2
                        ].copy()
                        img_slice[img_slice == self.__current_page_erase_color] = 0
                        img_slice[img_slice != 0] = 255
                        # Maybe next time in others PDF pages complexest condition raisin
                        # But in current time i dont consider these
                        if 5 < len(np.unique(np.nonzero(img_slice)[1])) >= \
                           (next_hline[0] - hline[2] - 2) / 2:
                            hrow.insert(
                                hline_index,
                                [
                                    hline[2], hline[3],
                                    next_hline[0], next_hline[1]
                                ]
                            )

            if not is_real_table:
                del tables[index]
            else:
                tables_data.append(data)
            index += 1

        n = 0
        for table in tables:
            # print('>', table)
            for row in table:
                for line in row:
                    n += 1
                    x0, y0, x1, y1 = line[:4]
                    gray = cv2.line(gray, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imwrite(
            os.path.splitext(self.pages_images[self.__current_page])[0] + '-out-gray.jpg', gray
        )


class PDFaParser(object):
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf = open(pdf_path)
        self.pages = []
        self.temp = tempfile.mkdtemp(prefix="pidifa_")
        print(self.temp + '/pages')

    def convert_pdf_to_docx(self):
        words, boxes = self.fold_pdf_style()

    def fold_pdf_style(self):
        # Extract XML
        tpath = os.path.join(self.temp, "pdf.xml")
        # print("----->", f"pdftohtml -c -xml {self.pdf_path} {tpath}")
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
#     print('No lines were found')
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

# print('number of Hough lines:', len(lines))

# filtered_lines = []

# if filter:
#     for i in range(len(lines)): # filtering
#         if line_flags[i]:
#             filtered_lines.append(lines[i])

#     print('Number of filtered lines:',  len(filtered_lines))
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
