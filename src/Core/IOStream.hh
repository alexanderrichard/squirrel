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

/*
 * internal classes of CompressedStream are based on
 *
 * ============================================================================
 * gzstream, C++ iostream classes wrapping the zlib compression library.
 * Copyright (C) 2001  Deepak Bandyopadhyay, Lutz Kettner
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 * ============================================================================
 *
 * File          : gzstream.h
 * Revision      : $Revision: 1.5 $
 * Revision_date : $Date: 2002/04/26 23:30:15 $
 * Author(s)     : Deepak Bandyopadhyay, Lutz Kettner
 *
 * Standard streambuf implementation following Nicolai Josuttis, "The
 * Standard C++ Library".
 * ============================================================================
 */

#ifndef CORE_IOSTREAM_HH_
#define CORE_IOSTREAM_HH_

#include <iostream>
#include <fstream>
#include <string.h>
#include <zlib.h>
#include "Types.hh"
#include "Utils.hh"

namespace Core {

/*
 * IOStream
 *
 * wrapper for file-IO operations
 */
class IOStream {
private:
	virtual char get() = 0;
public:
	IOStream();
	virtual ~IOStream();

	virtual void open(const std::string& filename, const std::ios_base::openmode mode) = 0;
	virtual bool is_open() = 0;
	virtual void close() = 0;
	virtual bool eof() = 0;

	static void endl(std::ostream& stream) { stream << std::endl; }
	static void scientific(std::ostream& stream) { stream << std::scientific; }
	virtual IOStream& operator<<(void (*fptr)(std::ostream&)) = 0;

	virtual IOStream& operator<<(u8) = 0;
	virtual IOStream& operator<<(u32) = 0;
	virtual IOStream& operator<<(u64) = 0;
	virtual IOStream& operator<<(s8) = 0;
	virtual IOStream& operator<<(s32) = 0;
	virtual IOStream& operator<<(s64) = 0;
	virtual IOStream& operator<<(f32) = 0;
	virtual IOStream& operator<<(f64) = 0;
	virtual IOStream& operator<<(bool) = 0;
	virtual IOStream& operator<<(char) = 0;
	virtual IOStream& operator<<(const char*) = 0;
	virtual IOStream& operator<<(const std::string&) = 0;

	virtual IOStream& operator>>(u8&) = 0;
	virtual IOStream& operator>>(u32&) = 0;
	virtual IOStream& operator>>(u64&) = 0;
	virtual IOStream& operator>>(s8&) = 0;
	virtual IOStream& operator>>(s32&) = 0;
	virtual IOStream& operator>>(s64&) = 0;
	virtual IOStream& operator>>(f32&) = 0;
	virtual IOStream& operator>>(f64&) = 0;
	virtual IOStream& operator>>(bool&) = 0;
	virtual IOStream& operator>>(char&) = 0;
	virtual IOStream& operator>>(std::string&); // reads a 0-terminated string
};


/*
 * BinaryStream
 */
class BinaryStream : public IOStream {
private:
	typedef IOStream Precursor;
	static const u64 bufferSize_;
	std::fstream stream_;
	u64 remainingBytes_;
	u64 unreadBufferedBytes_;
	u64 bufferPointer_;
	char* buffer_;
private:
	virtual void readBuffer();
	virtual char get();
	virtual void get(char* data, u32 size);
public:
	using Precursor::operator>>;
public:
	BinaryStream();
	BinaryStream(const std::string& filename, const std::ios_base::openmode mode);
	virtual ~BinaryStream();

	virtual void open(const std::string& filename, const std::ios_base::openmode mode);
	virtual bool is_open();
	virtual void close();
	virtual bool eof();

	virtual IOStream& operator<<(void (*fptr)(std::ostream&));

	virtual IOStream& operator<<(u8);
	virtual IOStream& operator<<(u32);
	virtual IOStream& operator<<(u64);
	virtual IOStream& operator<<(s8);
	virtual IOStream& operator<<(s32);
	virtual IOStream& operator<<(s64);
	virtual IOStream& operator<<(f32);
	virtual IOStream& operator<<(f64);
	virtual IOStream& operator<<(bool);
	virtual IOStream& operator<<(char);
	virtual IOStream& operator<<(const char*);
	virtual IOStream& operator<<(const std::string&);

	virtual IOStream& operator>>(u8&);
	virtual IOStream& operator>>(u32&);
	virtual IOStream& operator>>(u64&);
	virtual IOStream& operator>>(s8&);
	virtual IOStream& operator>>(s32&);
	virtual IOStream& operator>>(s64&);
	virtual IOStream& operator>>(f32&);
	virtual IOStream& operator>>(f64&);
	virtual IOStream& operator>>(bool&);
	virtual IOStream& operator>>(char&);
};

/*
 * AsciiStream
 */
class AsciiStream : public IOStream {
private:
	typedef IOStream Precursor;
	std::fstream stream_;
private:
	virtual char get();
public:
	using Precursor::operator>>;
public:
	AsciiStream() {}
	AsciiStream(const std::string& filename, const std::ios_base::openmode mode);
	virtual ~AsciiStream();

	virtual void open(const std::string& filename, const std::ios_base::openmode mode);
	virtual bool is_open();
	virtual void close();
	virtual bool eof();

	virtual IOStream& operator<<(void (*fptr)(std::ostream&));

	virtual IOStream& operator<<(u8);
	virtual IOStream& operator<<(u32);
	virtual IOStream& operator<<(u64);
	virtual IOStream& operator<<(s8);
	virtual IOStream& operator<<(s32);
	virtual IOStream& operator<<(s64);
	virtual IOStream& operator<<(f32);
	virtual IOStream& operator<<(f64);
	virtual IOStream& operator<<(bool);
	virtual IOStream& operator<<(char);
	virtual IOStream& operator<<(const char*);
	virtual IOStream& operator<<(const std::string&);

	virtual IOStream& operator>>(u8&);
	virtual IOStream& operator>>(u32&);
	virtual IOStream& operator>>(u64&);
	virtual IOStream& operator>>(s8&);
	virtual IOStream& operator>>(s32&);
	virtual IOStream& operator>>(s64&);
	virtual IOStream& operator>>(f32&);
	virtual IOStream& operator>>(f64&);
	virtual IOStream& operator>>(bool&);
	virtual IOStream& operator>>(char&);
};

/*
 * CompressedStream
 */
class CompressedStream : public IOStream {
private:
	/* ------------------------------------------------------------------------- */
	/*
	 * Internal classes to implement gzstream. See below for user classes.
	 */
	/* ------------------------------------------------------------------------- */

	class gzstreambuf : public std::streambuf {
	private:
		static const int bufferSize = 47+256;    // size of data buff
		// totals 512 bytes under g++ for igzstream at the end.

		gzFile           file;               // file handle for compressed file
		char             buffer[bufferSize]; // data buffer
		char             opened;             // open/close state of stream
		int              mode;               // I/O mode

		int flush_buffer();
	public:
		gzstreambuf() :
			file(0), opened(0), mode(0) {
			setp( buffer, buffer + (bufferSize-1));
			setg( buffer + 4,     // beginning of putback area
					buffer + 4,     // read position
					buffer + 4);    // end position
			// ASSERT: both input & output capabilities will not be used together
		}
		int is_open() { return opened; }
		gzstreambuf* open( const char* name, int open_mode);
		gzstreambuf* close();
		~gzstreambuf() { close(); }

		virtual int     overflow( int c = EOF);
		virtual int     underflow();
		virtual int     sync();
	};

	class gzstreambase : virtual public std::ios {
	protected:
		gzstreambuf buf;
	public:
		gzstreambase() { init(&buf); }
		gzstreambase( const char* name, int open_mode);
		~gzstreambase();
		void open( const char* name, int open_mode);
		void close();
		gzstreambuf* rdbuf() { return &buf; }
		bool is_open() { return rdbuf()->is_open() && good(); }
	};

	/* ------------------------------------------------------------------------- */
	/*
	 * User classes. Use igzstream and ogzstream analogously to ifstream and
	 * ofstream respectively. They read and write files based on the gz*
	 * function interface of the zlib. Files are compatible with gzip compression.
	 */
	/* ------------------------------------------------------------------------- */

	class igzstream : public gzstreambase, public std::istream {
	public:
		igzstream() : std::istream( &buf) {}
		igzstream( const char* name, int open_mode = std::ios::in)
		: gzstreambase( name, open_mode), std::istream( &buf) {}
		gzstreambuf* rdbuf() { return gzstreambase::rdbuf(); }
		void open( const char* name, int open_mode = std::ios::in) {
			gzstreambase::open( name, open_mode);
		}
	};

	class ogzstream : public gzstreambase, public std::ostream {
	public:
		ogzstream() : std::ostream( &buf) {}
		ogzstream( const char* name, int mode = std::ios::out)
		: gzstreambase( name, mode), std::ostream( &buf) {}
		gzstreambuf* rdbuf() { return gzstreambase::rdbuf(); }
		void open( const char* name, int open_mode = std::ios::out) {
			gzstreambase::open( name, open_mode);
		}
	};

	/* ------------------------------------------------------------------------- */

private:
	typedef IOStream Precursor;
	igzstream in_;
	ogzstream out_;
	std::ios::openmode mode_;
private:
	virtual char get();
public:
	using Precursor::operator>>;
public:
	CompressedStream();
	CompressedStream(const std::string& filename, const std::ios_base::openmode mode);
	virtual ~CompressedStream();

	virtual void open(const std::string& filename, const std::ios_base::openmode mode);
	virtual bool is_open();
	virtual void close();
	virtual bool eof();

	virtual IOStream& operator<<(void (*fptr)(std::ostream&));

	virtual IOStream& operator<<(u8);
	virtual IOStream& operator<<(u32);
	virtual IOStream& operator<<(u64);
	virtual IOStream& operator<<(s8);
	virtual IOStream& operator<<(s32);
	virtual IOStream& operator<<(s64);
	virtual IOStream& operator<<(f32);
	virtual IOStream& operator<<(f64);
	virtual IOStream& operator<<(bool);
	virtual IOStream& operator<<(char);
	virtual IOStream& operator<<(const char*);
	virtual IOStream& operator<<(const std::string&);

	virtual IOStream& operator>>(u8&);
	virtual IOStream& operator>>(u32&);
	virtual IOStream& operator>>(u64&);
	virtual IOStream& operator>>(s8&);
	virtual IOStream& operator>>(s32&);
	virtual IOStream& operator>>(s64&);
	virtual IOStream& operator>>(f32&);
	virtual IOStream& operator>>(f64&);
	virtual IOStream& operator>>(bool&);
	virtual IOStream& operator>>(char&);
};

} // namespace

#endif /* CORE_IOSTREAM_HH_ */
