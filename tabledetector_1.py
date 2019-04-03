import os
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
            cv2.imwrite(
                os.path.splitext(self.pages_images[self.__current_page])[0] + '-out.jpg',
                self.__current_page_image
            )
            # detect table lines
            gray = cv2.cvtColor(self.__current_page_image, cv2.COLOR_BGR2GRAY)
            lsd = cv2.createLineSegmentDetector(0)
            index1 = 0
            lines = lsd.detect(gray)[0]
            final_lines = [
                [int(round(p[0])), int(round(p[1])), int(round(p[2])), int(round(p[3]))]
                for p in (line[0] for line in lines)
            ]
            final_lines = sorted(final_lines, key=lambda x: [x[1], x[0]])
            # for index in range(len(final_lines)):
            #     x1, y1, x2, y2 = final_lines[index]
            #     if x1 != x2 and abs(x1 - x2) <= 5:
            #         final_lines[index][0] = max(x1, x2)
            #         final_lines[index][2] = max(x1, x2)
            #     elif y1 != y2 and abs(y1 - y2) <= 5:
            #         final_lines[index][1] = max(y1, y2)
            #         final_lines[index][3] = max(y1, y2)

            # oldx = -1
            oldy = -1
            xs = set()
            ys = set()
            tables_y = []
            table = {}
            tables = []
            while index1 < len(final_lines):
                line1 = final_lines[index1]
                x1, y1, x2, y2 = line1

                # Find tables in vertical arrows
                new_table = (len(tables_y) == 0 or (y1 - tables_y[-1]) >= 5) and y1 not in tables_y
                for item in ys:
                    if not new_table:
                        break
                    if (len(ys) and (y1 - item) < 5):
                        new_table = False
                if new_table:
                    table = []
                    tables.append(table)
                    tables_y.append(y1)
                    gray = cv2.putText(
                        gray, str(len(tables_y)), (5, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA
                    )

                xs.add(x1)
                xs.add(x2)
                ys.add(y1)
                ys.add(y2)
                if oldy != y1:
                    # print(y1)
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

                index2 = 0
                deleted = False
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
                    else:
                        diff = 0

                    if diff and diff < 5:
                        del final_lines[index2]
                        deleted = True
                    else:
                        index2 += 1

                if line1 not in table:
                    table.append(line1)

                if not deleted:
                    index1 += 1

            # for x1, y1, x2, y2 in final_lines:
            #     for x in xs:
            #         pass

            #     for y in ys:
            #         pass
            print(*tables_y)
            print(tables)

            for line in final_lines:
                x0, y0, x1, y1 = line
                gray = cv2.line(gray, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(
                os.path.splitext(self.pages_images[self.__current_page])[0] + '-out-gray.jpg', gray
            )

        self.__hierarchy.pop()
        self.__current_tag = self.__hierarchy[-1][0] if len(self.__hierarchy) else None
        self.__current_attrs = self.__hierarchy[-1][1] if len(self.__hierarchy) else {}


class PDFaParser(object):
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf = open(pdf_path)
        self.pages = []
        self.temp = tempfile.mkdtemp(prefix="pidifa_")

    def convert_pdf_to_docx(self):
        words, boxes = self.fold_pdf_style()

    def fold_pdf_style(self):
        # Extract XML
        tpath = os.path.join(self.temp, "pdf.xml")
        print("----->", f"pdftohtml -c -xml {self.pdf_path} {tpath}")
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


pdfp = PDFaParser('/home/mohsen/lab/pdfaSnippets/2.pdf')
pdfp.convert_pdf_to_docx()
pdfp





file_path = '2.jpg'
img = cv2.imread(file_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 90, 150, apertureSize=3)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
kernel = np.ones((5, 5), np.uint8)
edges = cv2.erode(edges, kernel, iterations=1)
cv2.imwrite('canny.jpg', edges)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

if not lines.any():
    print('No lines were found')
    exit()

if filter:
    rho_threshold = 15
    theta_threshold = 0.1

    # how many lines are similar to a given one
    similar_lines = {i : [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x : len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
            continue

        for j in range(i + 1,  len(lines)): # we are only considering those elements that had less similar line
            if not line_flags[indices[j]]: # and only if we have not disregarded them already
                continue

            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

print('number of Hough lines:',  len(lines))

filtered_lines = []

if filter:
    for i in range(len(lines)): # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    print('Number of filtered lines:',  len(filtered_lines))
else:
    filtered_lines = lines

for line in filtered_lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('hough.jpg', img)

