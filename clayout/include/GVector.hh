#ifndef _GVECTOR_H_
#define _GVECTOR_H_


// An arbitrary dimension vector class.
// GVector<N> can be constructed either from a std::array<double,N>
//   or by passing in N doubles as arguments to the constructor.

#include <type_traits>
#include <array>
#include <ostream>
#include <cmath>

template<unsigned int N>
struct GVector{
  std::array<double,N> data;

  GVector(std::array<double,N> arr) : data(arr) {}

  template<typename... Args>
  GVector(Args... args) : data{double(args)...} {
    static_assert(sizeof...(args) == N,
                  "Arguments passed do not match templated size");
  }

  GVector(){
    data.fill(0);
  }

  double Mag2() const;

  double Mag() const {
    return std::sqrt(Mag2());
  }

  GVector<N> UnitVector() const{
    return (*this)/Mag();
  }

  bool operator==(const GVector<N>& other) const {
    return data == other.data;
  }

  // Modification operators
  GVector<N>& operator+=(const GVector<N>& other){
    for(unsigned int i=0; i<N; i++){
      data[i] += other.data[i];
    }
    return *this;
  }

  GVector<N>& operator-=(const GVector<N>& other){
    for(unsigned int i=0; i<N; i++){
      data[i] -= other.data[i];
    }
    return *this;
  }

  GVector<N>& operator*=(const double other){
    for(unsigned int i=0; i<N; i++){
      data[i] *= other;
    }
    return *this;
  }

  GVector<N>& operator/=(const double other){
    for(unsigned int i=0; i<N; i++){
      data[i] /= other;
    }
    return *this;
  }

  // Cross-product
  GVector<N> Cross(const GVector<N>& b) const {
    static_assert(N==3,"Cross product available only for dimensions == 3");
    return GVector<N>(Y()*b.Z() - Z()*b.Y(),
                      Z()*b.X() - X()*b.Z(),
                      X()*b.Y() - Y()*b.X());
  }

  GVector<N> operator^(const GVector<N>& b) const {
    static_assert(N==3,"Cross product available only for dimensions == 3");
    return Cross(b);
  }

  // Dot-product
  double Dot(const GVector<N>& b) const {
    double output = 0;
    for(unsigned int i=0; i<N; i++){
      output += data[i]*b.data[i];
    }
    return output;
  }

  // Ease of use accessors for dimensions x,y,z
  double& X() {
    static_assert(N>=1,"X available only for dimensions >= 1");
    return data[0];
  }

  const double& X() const {
    static_assert(N>=1,"X available only for dimensions >= 1");
    return data[0];
  }

  double& Y() {
    static_assert(N>=2,"Y available only for dimensions >= 2");
    return data[1];
  }

  const double& Y() const {
    static_assert(N>=2,"Y available only for dimensions >= 2");
    return data[1];
  }

  double& Z() {
    static_assert(N>=3,"Z available only for dimensions >= 3");
    return data[2];
  }

  const double& Z() const {
    static_assert(N>=3,"Z available only for dimensions >= 3");
    return data[2];
  }
};

// Vector addition
template<unsigned int N>
GVector<N> operator+(GVector<N> a, const GVector<N>& b){
  return a += b;
}

// Vector subtraction
template<unsigned int N>
GVector<N> operator-(GVector<N> a, const GVector<N>& b){
  return a -= b;
}

// Left scalar multiplication
template<unsigned int N>
GVector<N> operator*(double a, GVector<N> b){
  return b *= a;
}

// Right scalar multiplication
template<unsigned int N>
GVector<N> operator*(GVector<N> a, double b){
  return a *= b;
}

// Right scalar division
template<unsigned int N>
GVector<N> operator/(GVector<N> a, double b){
  return a /= b;
}

// Dot product
template<unsigned int N>
double operator*(const GVector<N>& a, const GVector<N>& b){
  return a.Dot(b);
}

template<unsigned int N>
double GVector<N>::Mag2() const {
  return (*this).Dot(*this);
}

template<unsigned int N>
std::ostream& operator<<(std::ostream& st, GVector<N> gv){
  st << "(";
  for(unsigned int i=0; i<N; i++){
    st << gv.data[i];
    if(i!=N-1){
      st << ", ";
    }
  }
  st << ")";
  return st;
}

#endif /* _GVECTOR_H_ */
