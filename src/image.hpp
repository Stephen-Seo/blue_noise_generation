#ifndef DITHERING_IMAGE_HPP
#define DITHERING_IMAGE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace image {
enum class color_type {
  Black,
  Red,
  Green,
  Blue,
  Alpha,
};

enum class file_type {
  PBM,
  PGM,
  PPM,
  PNG,
};

class Base {
 public:
  Base() = default;
  virtual ~Base() {}

  Base(const Base &other) = default;
  Base(Base &&other) = default;

  Base &operator=(const Base &other) = default;
  Base &operator=(Base &&other) = default;

  virtual void randomize() = 0;

  virtual unsigned int getSize() const = 0;
  virtual uint8_t *getData() = 0;
  virtual const uint8_t *getDataC() const = 0;

  virtual unsigned int getWidth() const = 0;
  virtual unsigned int getHeight() const = 0;

  virtual int getTypesCount() = 0;
  virtual std::vector<color_type> getTypes() = 0;
  virtual int getTypeStride(color_type type) = 0;

  virtual bool canWriteFile(file_type type) = 0;
  virtual bool writeToFile(file_type type, bool canOverwrite,
                           const char *filename) = 0;
  virtual bool writeToFile(file_type type, bool canOverwrite,
                           const std::string &filename) = 0;
  bool isValid() const;
};

class Bl : public Base {
 public:
  Bl();
  Bl(int width, int height);
  Bl(const std::vector<uint8_t> &data, int width);
  Bl(std::vector<uint8_t> &&data, int width);
  Bl(const std::vector<float> &data, int width);
  virtual ~Bl() {}

  Bl(const Bl &other) = default;
  Bl(Bl &&other) = default;

  Bl &operator=(const Bl &other) = default;
  Bl &operator=(Bl &&other) = default;

  void randomize() override;

  unsigned int getSize() const override;
  uint8_t *getData() override;
  const uint8_t *getDataC() const override;

  unsigned int getWidth() const override;
  unsigned int getHeight() const override;

  int getTypesCount() override { return 1; }
  std::vector<color_type> getTypes() override { return {color_type::Black}; }
  int getTypeStride(color_type) override { return 0; }

  bool canWriteFile(file_type type) override;
  bool writeToFile(file_type type, bool canOverwrite,
                   const char *filename) override;
  bool writeToFile(file_type type, bool canOverwrite,
                   const std::string &filename) override;

 private:
  std::vector<uint8_t> data;
  int width;
  int height;
};
}  // namespace image

#endif
