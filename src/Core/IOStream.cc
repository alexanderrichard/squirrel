/*
 * Copyright 2016 Alexander Richard
 *
 * This file is part of Squirrel.
 *
 * Licensed under the Academic Free License 3.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should have received a copy of the License along with Squirrel.
 * If not, see <https://opensource.org/licenses/AFL-3.0>.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "IOStream.hh"
#include <stdlib.h>

using namespace Core;

IOStream::IOStream()
{}

IOStream::~IOStream()
{}

IOStream& IOStream::operator>>(std::string& str) {
	str.clear();
	char c;
	c = get();
	while (c != 0) {
		str.push_back(c);
		c = get();
	}
	return *this;
}

/* ------------------------------------------------------------------------- */
/*
 * BinaryStream
 */
/* ------------------------------------------------------------------------- */
const u64 BinaryStream::bufferSize_ = 4096;

BinaryStream::BinaryStream() :
		remainingBytes_(0),
		unreadBufferedBytes_(0),
		bufferPointer_(bufferSize_),
		buffer_(new char[bufferSize_])
{}

BinaryStream::BinaryStream(const std::string& filename, const std::ios_base::openmode mode) :
		remainingBytes_(0),
		unreadBufferedBytes_(0),
		bufferPointer_(bufferSize_),
		buffer_(new char[bufferSize_])
{
	open(filename, mode);
}

void BinaryStream::open(const std::string& filename, const std::ios_base::openmode mode) {
	stream_.open(filename.c_str(), mode | std::ios::binary);
	if (!stream_.is_open()) {
		std::cerr << "Failed to open binary file " << filename << ". Abort." << std::endl;
		exit(1);
	}
	stream_.seekg(0, stream_.end);
    remainingBytes_ = stream_.tellg();
    stream_.seekg(0, stream_.beg);
    bufferPointer_ = bufferSize_;
    unreadBufferedBytes_ = 0;
}

BinaryStream::~BinaryStream() {
	if (is_open()) {
		close();
	}
	delete buffer_;
}

bool BinaryStream::is_open() {
	return stream_.is_open();
}

void BinaryStream::close() {
	stream_.close();
}

bool BinaryStream::eof() {
	return stream_.eof();
}

void BinaryStream::readBuffer() {
	if (remainingBytes_ == 0) {
		std::cout << "Error: BinaryStream::readBuffer: No unread bytes left in file. Abort." << std::endl;
		exit(1);
	}
	u32 bytesToRead = std::min(remainingBytes_, bufferSize_);
	stream_.read(buffer_, sizeof(char) * bytesToRead);
	for (u64 i = bytesToRead; i < bufferSize_; i++) {
		buffer_[i] = 0;
	}
	remainingBytes_ -= bytesToRead;
	bufferPointer_ = 0;
	unreadBufferedBytes_ = bytesToRead;
}

char BinaryStream::get() {
	if (unreadBufferedBytes_ == 0)
		readBuffer();
	char c = buffer_[bufferPointer_];
	bufferPointer_++;
	unreadBufferedBytes_--;
	return c;
}

void BinaryStream::get(char* data, u32 size) {
	for (u32 i = 0; i < size; i++) {
		if (unreadBufferedBytes_ == 0)
			readBuffer();
		data[i] = buffer_[bufferPointer_];
		bufferPointer_++;
		unreadBufferedBytes_--;
	}
}

IOStream& BinaryStream::operator<<(void (*fptr)(std::ostream&)) { fptr(stream_); return *this; }

IOStream& BinaryStream::operator<<(u8 n) { stream_.write((const char*) &n, sizeof(u8)); return *this; }
IOStream& BinaryStream::operator<<(u32 n) { stream_.write((const char*) &n, sizeof(u32)); return *this; }
IOStream& BinaryStream::operator<<(u64 n) { stream_.write((const char*) &n, sizeof(u64)); return *this; }
IOStream& BinaryStream::operator<<(s8 n) { stream_.write((const char*) &n, sizeof(s8)); return *this; }
IOStream& BinaryStream::operator<<(s32 n) { stream_.write((const char*) &n, sizeof(s32)); return *this; }
IOStream& BinaryStream::operator<<(s64 n) { stream_.write((const char*) &n, sizeof(s64)); return *this; }
IOStream& BinaryStream::operator<<(f32 n) { stream_.write((const char*) &n, sizeof(f32)); return *this; }
IOStream& BinaryStream::operator<<(f64 n) { stream_.write((const char*) &n, sizeof(f64)); return *this; }
IOStream& BinaryStream::operator<<(bool n) { stream_.write((const char*) &n, sizeof(bool)); return *this; }
IOStream& BinaryStream::operator<<(char n) { stream_.write((const char*) &n, sizeof(char)); return *this; }
IOStream& BinaryStream::operator<<(const char* n) {
	std::string tmp(n);
	stream_.write(n, sizeof(char) * tmp.size());
	return *this;
}
IOStream& BinaryStream::operator<<(const std::string& n) {
	stream_.write((const char*) n.c_str(), sizeof(char) * n.size());
	return *this;
}

IOStream& BinaryStream::operator>>(u8& n) { get((char*) &n, sizeof(u8)); return *this; }
IOStream& BinaryStream::operator>>(u32& n) { get((char*) &n, sizeof(u32)); return *this; }
IOStream& BinaryStream::operator>>(u64& n) { get((char*) &n, sizeof(u64)); return *this; }
IOStream& BinaryStream::operator>>(s8& n) { get((char*) &n, sizeof(s8)); return *this; }
IOStream& BinaryStream::operator>>(s32& n) { get((char*) &n, sizeof(s32)); return *this; }
IOStream& BinaryStream::operator>>(s64& n) { get((char*) &n, sizeof(s64)); return *this; }
IOStream& BinaryStream::operator>>(f32& n) { get((char*) &n, sizeof(f32)); return *this; }
IOStream& BinaryStream::operator>>(f64& n) { get((char*) &n, sizeof(f64)); return *this; }
IOStream& BinaryStream::operator>>(bool& n) { get((char*) &n, sizeof(bool)); return *this; }
IOStream& BinaryStream::operator>>(char& n) { get((char*) &n, sizeof(char)); return *this; }

/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*
 * AsciiStream
 */
/* ------------------------------------------------------------------------- */

AsciiStream::AsciiStream(const std::string& filename, const std::ios_base::openmode mode) {
	open(filename, mode);
}

void AsciiStream::open(const std::string& filename, const std::ios_base::openmode mode) {
	stream_.open(filename.c_str(), mode);
	if (!stream_.is_open()) {
		std::cerr << "Failed to open file " << filename << ". Abort." << std::endl;
		exit(1);
	}
}

AsciiStream::~AsciiStream() {
	if (is_open()) {
		close();
	}
}

bool AsciiStream::is_open() {
	return stream_.is_open();
}

void AsciiStream::close() {
	stream_.close();
}

bool AsciiStream::eof() {
	return stream_.eof();
}

char AsciiStream::get() {
	char c;
	stream_.get(c);
	return c;
}

IOStream& AsciiStream::operator<<(void (*fptr)(std::ostream&)) { fptr(stream_); return *this; }

IOStream& AsciiStream::operator<<(u8 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(u32 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(u64 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(s8 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(s32 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(s64 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(f32 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(f64 n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(bool n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(char n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(const char* n) { stream_ << n; return *this; }
IOStream& AsciiStream::operator<<(const std::string& n) { stream_ << n; return *this; }

IOStream& AsciiStream::operator>>(u8& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(u32& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(u64& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(s8& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(s32& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(s64& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(f32& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(f64& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(bool& n) { stream_ >> n; return *this; }
IOStream& AsciiStream::operator>>(char& n) { stream_ >> n; return *this; }

/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*
 * CompressedStream
 */
/* ------------------------------------------------------------------------- */

CompressedStream::CompressedStream() :
		mode_(std::ios::out)
{}

CompressedStream::CompressedStream(const std::string& filename, const std::ios_base::openmode mode) :
		mode_(mode)
{
	std::string fn(filename);
	if (mode_ == std::ios::out) {
		if (fn.substr(fn.length() - 3).compare(".gz") != 0)
			fn.append(".gz");
	}
	open(fn, mode);
}

void CompressedStream::open(const std::string& filename, const std::ios_base::openmode mode)
{
	mode_ = mode;
	if (mode == std::ios::in) {
		in_.open(filename.c_str(), mode);
		if (!in_.is_open()) {
			std::cerr << "Failed to open file " << filename << ". Abort." << std::endl;
			exit(1);
		}
	}
	else {
		std::string fn(filename);
		if (fn.substr(fn.length() - 3).compare(".gz") != 0)
			fn.append(".gz");
		out_.open(fn.c_str(), mode);
		if (!out_.is_open()) {
			std::cerr << "Failed to open file " << fn << ". Abort." << std::endl;
			exit(1);
		}
	}
}

CompressedStream::~CompressedStream() {
	if (in_.is_open()) {
		in_.close();
	}
	if (out_.is_open()) {
		out_.close();
	}
}

bool CompressedStream::is_open() {
	if (mode_ == std::ios::in) {
		return in_.is_open();
	}
	else {
		return out_.is_open();
	}
}

void CompressedStream::close() {
	if (in_.is_open()) {
		in_.close();
	}
	if (out_.is_open()) {
		out_.close();
	}
}

bool CompressedStream::eof() {
	if (in_.is_open()) {
		return in_.eof();
	}
	if (out_.is_open()) {
		return out_.eof();
	}
	else {
		return true;
	}
}

char CompressedStream::get() {
	char c;
	in_.get(c);
	return c;
}

IOStream& CompressedStream::operator<<(void (*fptr)(std::ostream&)) { fptr(out_); return *this; }

IOStream& CompressedStream::operator<<(u8 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(u32 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(u64 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(s8 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(s32 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(s64 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(f32 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(f64 n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(bool n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(char n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(const char* n) { out_ << n; return *this; }
IOStream& CompressedStream::operator<<(const std::string& n) { out_ << n; return *this; }

IOStream& CompressedStream::operator>>(u8& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(u32& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(u64& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(s8& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(s32& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(s64& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(f32& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(f64& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(bool& n) { in_ >> n; return *this; }
IOStream& CompressedStream::operator>>(char& n) { in_ >> n; return *this; }

/* ------------------------------------------------------------------------- */
// ----------------------------------------------------------------------------
// Internal classes to implement gzstream. See header file for user classes.
// ----------------------------------------------------------------------------

// --------------------------------------
// class gzstreambuf:
// --------------------------------------

CompressedStream::gzstreambuf* CompressedStream::gzstreambuf::open( const char* name, int open_mode) {
	if ( is_open())
		return (gzstreambuf*)0;
	mode = open_mode;
	// no append nor read/write mode
	if ((mode & std::ios::ate) || (mode & std::ios::app)
			|| ((mode & std::ios::in) && (mode & std::ios::out)))
		return (gzstreambuf*)0;
	char  fmode[10];
	char* fmodeptr = fmode;
	if ( mode & std::ios::in)
		*fmodeptr++ = 'r';
	else if ( mode & std::ios::out)
		*fmodeptr++ = 'w';
	*fmodeptr++ = 'b';
	*fmodeptr = '\0';
	file = gzopen( name, fmode);
	if (file == 0)
		return (gzstreambuf*)0;
	opened = 1;
	return this;
}

CompressedStream::gzstreambuf * CompressedStream::gzstreambuf::close() {
	if ( is_open()) {
		sync();
		opened = 0;
		if ( gzclose( file) == Z_OK)
			return this;
	}
	return (gzstreambuf*)0;
}

int CompressedStream::gzstreambuf::underflow() { // used for input buffer only
	if ( gptr() && ( gptr() < egptr()))
		return * reinterpret_cast<unsigned char *>( gptr());

	if ( ! (mode & std::ios::in) || ! opened)
		return EOF;
	// Josuttis' implementation of inbuf
	int n_putback = gptr() - eback();
	if ( n_putback > 4)
		n_putback = 4;
	memcpy( buffer + (4 - n_putback), gptr() - n_putback, n_putback);

	int num = gzread( file, buffer+4, bufferSize-4);
	if (num <= 0) // ERROR or EOF
		return EOF;

	// reset buffer pointers
	setg( buffer + (4 - n_putback),   // beginning of putback area
			buffer + 4,                 // read position
			buffer + 4 + num);          // end of buffer

	// return next character
	return * reinterpret_cast<unsigned char *>( gptr());
}

int CompressedStream::gzstreambuf::flush_buffer() {
	// Separate the writing of the buffer from overflow() and
	// sync() operation.
	int w = pptr() - pbase();
	if ( gzwrite( file, pbase(), w) != w)
		return EOF;
	pbump( -w);
	return w;
}

int CompressedStream::gzstreambuf::overflow( int c) { // used for output buffer only
	if ( ! ( mode & std::ios::out) || ! opened)
		return EOF;
	if (c != EOF) {
		*pptr() = c;
		pbump(1);
	}
	if ( flush_buffer() == EOF)
		return EOF;
	return c;
}

int CompressedStream::gzstreambuf::sync() {
	// Changed to use flush_buffer() instead of overflow( EOF)
	// which caused improper behavior with std::endl and flush(),
	// bug reported by Vincent Ricard.
	if ( pptr() && pptr() > pbase()) {
		if ( flush_buffer() == EOF)
			return -1;
	}
	return 0;
}

// --------------------------------------
// class gzstreambase:
// --------------------------------------

CompressedStream::gzstreambase::gzstreambase( const char* name, int mode) {
	init( &buf);
	open( name, mode);
}

CompressedStream::gzstreambase::~gzstreambase() {
	buf.close();
}

void CompressedStream::gzstreambase::open( const char* name, int open_mode) {
	if ( ! buf.open( name, open_mode))
		clear( rdstate() | std::ios::badbit);
}

void CompressedStream::gzstreambase::close() {
	if ( buf.is_open())
		if ( ! buf.close())
			clear( rdstate() | std::ios::badbit);
}

/* ------------------------------------------------------------------------- */
