#ifndef UTILS_H
#define UTILS_H

inline long read_long(FILE* fh) {
  long value;
  size_t result1 = fread(&value, sizeof(long), 1, fh);
  if (result1 != 1) {
    fprintf( stderr, "File reading error for a long. File is corrupt.\n");
    exit(1);
  }
  return value;
}

#endif /* UTILS_H */
